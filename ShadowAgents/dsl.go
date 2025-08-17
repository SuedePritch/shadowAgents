
package shadowagents

import (
	"context"
	"fmt"
	"log"
	"os"
	"strings"

	"github.com/google/generative-ai-go/genai"
	"google.golang.org/api/option"
)

// Global client instance. This allows all agents to share a single client.
var client *genai.Client

// Tool represents a function the agent can call.
// It includes a name, a description, and the function itself.
type Tool struct {
	Name        string
	Description string
	Function    func(context.Context, map[string]interface{}) (interface{}, error)
}

// AgentConfig holds the configuration for a single agent.
// This is the primary way a user will define a ShadowAgent.
type AgentConfig struct {
	Name         string `json:"name"`
	ModelName    string `json:"model_name"`
	SystemPrompt string `json:"system_prompt"`
}

// Agent is the core struct representing our intelligent agent.
// It encapsulates the Gemini model and all the agent's logic.
type Agent struct {
	name         string
	model        *genai.GenerativeModel
	tools        map[string]Tool
	systemPrompt string
}

// Setup initializes the global Gemini client. This should be called once at the
// beginning of the application's lifecycle.
func Setup(ctx context.Context, apiKey string) error {
	var err error
	client, err = genai.NewClient(ctx, option.WithAPIKey(apiKey))
	if err != nil {
		return fmt.Errorf("failed to create Gemini client: %w", err)
	}
	log.Println("Gemini client initialized successfully.")
	return nil
}

// Close should be called to close the global client when the application exits.
func Close() error {
	if client != nil {
		return client.Close()
	}
	return nil
}

// NewAgent creates and initializes a new Agent instance.
// It assumes Setup has already been called and uses the global client.
func NewAgent(ctx context.Context, config AgentConfig) (*Agent, error) {
	if client == nil {
		return nil, fmt.Errorf("Gemini client has not been set up. Call shadowagents.Setup first.")
	}

	// 1. Get the model instance from the shared client.
	model := client.GenerativeModel(config.ModelName)

	// 2. Initialize the tools map. Tools will be registered separately.
	toolsMap := make(map[string]Tool)
	
	// 3. Set the system instruction. The API for this has changed.
	model.SetSystemInstruction(genai.NewSystemInstruction(genai.NewTextPart(config.SystemPrompt)))

	return &Agent{
		name:         config.Name,
		model:        model,
		tools:        toolsMap,
		systemPrompt: config.SystemPrompt,
	}, nil
}

// RegisterTools adds one or more tools to an agent. This is a new method that
// allows tools to be defined and registered outside of the initial agent creation.
func (a *Agent) RegisterTools(tools []Tool) {
	var modelTools []*genai.Tool
	for _, tool := range tools {
		a.tools[tool.Name] = tool
		modelTools = append(modelTools, genai.NewTool(tool.Name, tool.Description))
	}
	a.model.SetTools(modelTools...)
	log.Printf("Registered %d tools to agent '%s'.", len(tools), a.name)
}

// Run executes the agent's logic for a given prompt.
// This method handles the conversational loop, including tool use.
func (a *Agent) Run(ctx context.Context, prompt string) (string, error) {
	// A simple chat session for a single turn. For multi-turn conversations,
	// you'd need to store and manage the history.
	cs := a.model.StartChat()
	
	// Send the user prompt to the model.
	resp, err := cs.SendMessage(ctx, genai.NewTextPart(prompt))
	if err != nil {
		return "", fmt.Errorf("failed to get model response: %w", err)
	}

	// Process the response, including tool calls.
	// This is a simple loop for a single tool call. More complex agentic
	// behavior might require a more sophisticated loop.
	for _, part := range resp.Candidates[0].Content.Parts {
		if fc, ok := part.(genai.FunctionCall); ok {
			tool, ok := a.tools[fc.Name]
			if !ok {
				return "", fmt.Errorf("tool '%s' not found", fc.Name)
			}
			
			// Call the tool's function with the provided arguments.
			result, err := tool.Function(ctx, fc.Args)
			if err != nil {
				return "", fmt.Errorf("failed to execute tool '%s': %w", fc.Name, err)
			}

			// Send the tool's result back to the model.
			resp, err := cs.SendMessage(ctx, genai.NewFunctionResponsePart(fc.Name, result))
			if err != nil {
				return "", fmt.Errorf("failed to send tool response to model: %w", err)
			}
			
			// The model should now provide a final response.
			if len(resp.Candidates) > 0 {
				return fmt.Sprint(resp.Candidates[0].Content.Parts[0]), nil
			}
		}
	}

	// If no tool was called, return the model's text response.
	if len(resp.Candidates) > 0 {
		if len(resp.Candidates[0].Content.Parts) > 0 {
			if textPart, ok := resp.Candidates[0].Content.Parts[0].(genai.TextPart); ok {
				return string(textPart), nil
			}
		}
	}

	return "No response from model.", nil
}

// NewSubAgentTool creates a tool that executes another agent.
// The tool's name and description are automatically generated from the sub-agent's
// own configuration.
func NewSubAgentTool(subAgent *Agent) Tool {
	// Generate a clean tool name from the sub-agent's name.
	toolName := strings.ReplaceAll(subAgent.name, " ", "")
	// Generate a descriptive string from the sub-agent's system prompt.
	toolDescription := fmt.Sprintf("Asks the %s agent to perform its function. %s", subAgent.name, subAgent.systemPrompt)

	return Tool{
		Name: toolName,
		Description: toolDescription,
		Function: func(ctx context.Context, args map[string]interface{}) (interface{}, error) {
			// Extract the prompt from the arguments to pass to the sub-agent.
			// This assumes the sub-agent's prompt is passed in an argument named "prompt".
			prompt, ok := args["prompt"].(string)
			if !ok {
				return nil, fmt.Errorf("missing or invalid 'prompt' argument for sub-agent")
			}
			log.Printf("Calling sub-agent '%s' with prompt: '%s'", subAgent.name, prompt)
			
			// Run the sub-agent and get its final response.
			response, err := subAgent.Run(ctx, prompt)
			if err != nil {
				return nil, fmt.Errorf("sub-agent run failed: %w", err)
			}
			// Return the sub-agent's response as the tool's result.
			return map[string]string{"result": response}, nil
		},
	}
}