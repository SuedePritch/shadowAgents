package shadowagents

import (
	"context"
	"fmt"
	"log"
	"strings"

	"github.com/google/generative-ai-go/genai"
	"google.golang.org/api/option"
)

// globalClient is the package-level Gemini client instance.
var globalClient *genai.Client

// AgentConfig holds the configuration for an agent.
type AgentConfig struct {
	Name        string
	Description string
	Model       string // e.g., "gemini-1.5-flash-latest"
}

// Agent represents an autonomous agent that can use tools.
type Agent struct {
	config AgentConfig
	model  *genai.GenerativeModel
	// tools is a map of tool names to their implementations.
	tools map[string]Tool
}

// Tool defines the interface for a tool that an agent can use.
// It consists of the schema for the model (FunctionDeclaration) and the
// actual Go function to execute.
type Tool struct {
	Schema  *genai.FunctionDeclaration
	Execute func(ctx context.Context, args map[string]any) (map[string]any, error)
}

// Config holds user-supplied Vertex AI setup options.
type Config struct {
	ProjectID string // required
	Location  string // required

	ModelName       string   // optional, e.g. "gemini-2.0-flash-001"
	Temperature     *float32 // optional
	TopP            *float32 // optional
	MaxOutputTokens *int32   // optional
}

var globalClient *genai.Client
var globalModel  *genai.GenerativeModel

// Setup initializes the Vertex AI client with required + optional settings.
func Setup(ctx context.Context, cfg Config) error {
	if cfg.ProjectID == "" || cfg.Location == "" {
		return fmt.Errorf("projectID and location are required")
	}
	if globalClient != nil {
		return nil // Already initialized
	}

	client, err := genai.NewClient(ctx, cfg.ProjectID, cfg.Location)
	if err != nil {
		return fmt.Errorf("failed to create vertex ai client: %w", err)
	}
	globalClient = client

	// If user specified a model, pre-configure it
	if cfg.ModelName != "" {
		m := client.GenerativeModel(cfg.ModelName)

		// Apply optional generation settings
		if cfg.Temperature != nil ||
			cfg.TopP != nil ||
			cfg.MaxOutputTokens != nil {

			m.GenerationConfig = &genai.GenerationConfig{}

			if cfg.Temperature != nil {
				m.GenerationConfig.Temperature = *cfg.Temperature
			}
			if cfg.TopP != nil {
				m.GenerationConfig.TopP = *cfg.TopP
			}
			if cfg.MaxOutputTokens != nil {
				m.GenerationConfig.MaxOutputTokens = *cfg.MaxOutputTokens
			}
		}

		globalModel = m
	}

	return nil
}

// Close shuts down the global Vertex AI client.
func Close() error {
	if globalClient != nil {
		return globalClient.Close()
	}
	return nil
}


// NewAgent creates and initializes a new Agent instance.
// It assumes Setup has already been called.
func NewAgent(config AgentConfig) (*Agent, error) {
	if globalClient == nil {
		return nil, fmt.Errorf("global client is not initialized; call Setup() first")
	}

	model := globalClient.GenerativeModel(config.Model)

	return &Agent{
		config: config,
		model:  model,
		tools:  make(map[string]Tool),
	}, nil
}

// RegisterTools adds one or more tools to the agent's capabilities.
func (a *Agent) RegisterTools(tools ...Tool) {
	for _, tool := range tools {
		if tool.Schema != nil {
			a.tools[tool.Schema.Name] = tool
		}
	}
}

// Run executes the agent's logic for a given prompt.
// It handles the conversation, including tool calls.
func (a *Agent) Run(ctx context.Context, prompt string) (string, error) {
	// Apply the agent's registered tools to the model for this run.
	// The tools must be set on the GenerativeModel, not the ChatSession.
	var genaiTools []*genai.Tool
	if len(a.tools) > 0 {
		var funcDecls []*genai.FunctionDeclaration
		for _, tool := range a.tools {
			funcDecls = append(funcDecls, tool.Schema)
		}
		genaiTools = append(genaiTools, &genai.Tool{FunctionDeclarations: funcDecls})
		a.model.Tools = genaiTools
	}

	session := a.model.StartChat()

	// Send the initial prompt to the model.
	resp, err := session.SendMessage(ctx, genai.Text(prompt))
	if err != nil {
		return "", fmt.Errorf("failed to send message: %w", err)
	}

	// The main loop to handle tool calls.
	for {
		if resp.Candidates == nil || len(resp.Candidates) == 0 || resp.Candidates[0].Content == nil {
			return responseToText(resp), nil // No valid response, return what we have.
		}

		// FIX: Check for a function call by performing a type assertion on the response Part.
		// The `GetFunctionCall` method does not exist directly on the `Part` interface.
		var fc *genai.FunctionCall
		for _, part := range resp.Candidates[0].Content.Parts {
			if f, ok := part.(genai.FunctionCall); ok {
				fc = &f
				break
			}
		}

		// If there is no function call, we have our final answer.
		if fc == nil {
			return responseToText(resp), nil
		}

		// The model has requested a tool call.
		tool, ok := a.tools[fc.Name]
		if !ok {
			return "", fmt.Errorf("model requested unknown tool: %s", fc.Name)
		}

		log.Printf("Agent '%s' is calling tool: %s with args: %v\n", a.config.Name, fc.Name, fc.Args)

		// Execute the tool.
		toolResult, err := tool.Execute(ctx, fc.Args)
		if err != nil {
			// Inform the model that the tool call failed.
			errorResponse := map[string]any{"error": err.Error()}
			resp, err = session.SendMessage(ctx, genai.FunctionResponse{Name: fc.Name, Response: errorResponse})
			if err != nil {
				return "", fmt.Errorf("failed to send tool error response: %w", err)
			}
			continue // Continue the loop to process the model's next step.
		}

		// Send the successful tool result back to the model.
		resp, err = session.SendMessage(ctx, genai.FunctionResponse{Name: fc.Name, Response: toolResult})
		if err != nil {
			return "", fmt.Errorf("failed to send tool result: %w", err)
		}
	}
}

// NewSubAgentTool creates a tool that executes another agent.
func NewSubAgentTool(subAgent *Agent) Tool {
	// Sanitize the agent name to be a valid function name.
	sanitizedName := strings.ReplaceAll(subAgent.config.Name, " ", "_")

	return Tool{
		Schema: &genai.FunctionDeclaration{
			Name:        sanitizedName,
			Description: subAgent.config.Description,
			Parameters: &genai.Schema{
				Type: genai.TypeObject,
				Properties: map[string]*genai.Schema{
					"prompt": {
						Type:        genai.TypeString,
						Description: "The detailed prompt or question to ask this agent.",
					},
				},
				Required: []string{"prompt"},
			},
		},
		Execute: func(ctx context.Context, args map[string]any) (map[string]any, error) {
			prompt, ok := args["prompt"].(string)
			if !ok {
				return nil, fmt.Errorf("invalid 'prompt' argument, expected a string")
			}

			// Run the sub-agent with the provided prompt.
			result, err := subAgent.Run(ctx, prompt)
			if err != nil {
				return nil, fmt.Errorf("sub-agent '%s' failed: %w", subAgent.config.Name, err)
			}

			// Return the result in a structured way for the parent agent.
			return map[string]any{"result": result}, nil
		},
	}
}

// responseToText is a helper to extract and concatenate text from a model's response.
func responseToText(resp *genai.GenerateContentResponse) string {
	var b strings.Builder
	if resp != nil && len(resp.Candidates) > 0 && resp.Candidates[0].Content != nil {
		for _, part := range resp.Candidates[0].Content.Parts {
			if txt, ok := part.(genai.Text); ok {
				b.WriteString(string(txt))
			}
		}
	}
	return b.String()
}