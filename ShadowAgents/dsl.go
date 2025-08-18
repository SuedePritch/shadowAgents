package shadowagents

import (
	"context"
	"encoding/json"
	"fmt"
	"log"

	"github.com/google/generative-ai-go/genai"
	"google.golang.org/api/option"
)

// Tool represents a function that an agent can execute.
// It takes a string argument (which can be JSON for complex inputs)
// and returns a string result and an error.
type Tool func(args string) (string, error)

// Agent represents an AI agent with a specific model, tools, and sub-agents.
type Agent struct {
	manager   *AgentManager // Reference to the central manager
	name      string
	model     *genai.GenerativeModel
	tools     map[string]*genai.Tool
	toolFuncs map[string]Tool // Go functions for the tools
	subAgents map[string]*Agent
}

// AgentManager is the central controller for creating and managing agents.
// It holds the connection to the AI service.
type AgentManager struct {
	client *genai.Client
	ctx    context.Context
	agents map[string]*Agent
}

// Setup initializes the AgentManager with a Gemini API key.
// This is the entry point for using the framework.
func Setup(ctx context.Context, apiKey string) (*AgentManager, error) {
	if apiKey == "" {
		return nil, fmt.Errorf("GEMINI_API_KEY environment variable not set")
	}

	client, err := genai.NewClient(ctx, option.WithAPIKey(apiKey))
	if err != nil {
		return nil, fmt.Errorf("failed to create genai client: %w", err)
	}

	return &AgentManager{
		client: client,
		ctx:    ctx,
		agents: make(map[string]*Agent),
	}, nil
}

// Close cleans up resources, primarily the connection to the Gemini API.
// It should be called when the application is shutting down.
func (m *AgentManager) Close() {
	if m.client != nil {
		m.client.Close()
		fmt.Println("AgentManager closed successfully.")
	}
}

// NewAgent creates a new agent instance and registers it with the manager.
// Each agent can have a unique name and system prompt to define its role.
func (m *AgentManager) NewAgent(name, systemPrompt string) *Agent {
	// Configure the generative model for the agent
	model := m.client.GenerativeModel("gemini-1.5-flash-latest")
	model.SystemInstruction = &genai.Content{
		Parts: []genai.Part{genai.Text(systemPrompt)},
	}
	model.Tools = []*genai.Tool{} // Initialize with no tools

	agent := &Agent{
		manager:   m,
		name:      name,
		model:     model,
		tools:     make(map[string]*genai.Tool),
		toolFuncs: make(map[string]Tool),
		subAgents: make(map[string]*Agent),
	}

	m.agents[name] = agent
	log.Printf("Agent '%s' created.", name)
	return agent
}

// RegisterTool adds a new tool that the agent can use.
// 'name': The function name the AI will call.
// 'description': A clear explanation of what the tool does for the AI.
// 'params': A JSON schema describing the tool's input parameters.
// 'toolFunc': The actual Go function to execute.
func (a *Agent) RegisterTool(name, description string, params *genai.Schema, toolFunc Tool) {
	if params == nil {
		// Create an empty schema if no params are needed
		params = &genai.Schema{Type: genai.TypeObject, Properties: map[string]*genai.Schema{}}
	}

	newTool := &genai.Tool{
		FunctionDeclarations: []*genai.FunctionDeclaration{{
			Name:        name,
			Description: description,
			Parameters:  params,
		}},
	}

	a.tools[name] = newTool
	a.toolFuncs[name] = toolFunc
	a.model.Tools = append(a.model.Tools, newTool) // Add the tool to the model's config

	log.Printf("Tool '%s' registered for agent '%s'.", name, a.name)
}

// NewSubagent creates a new agent that is a child of the current agent.
// This is useful for creating hierarchical agent structures.
func (a *Agent) NewSubagent(name, systemPrompt string) *Agent {
	// The sub-agent is created using the same manager but can have its own identity.
	subAgent := a.manager.NewAgent(name, systemPrompt)
	a.subAgents[name] = subAgent
	log.Printf("Sub-agent '%s' created under agent '%s'.", name, a.name)
	return subAgent
}

// ExecuteTask sends a prompt to the agent and handles the response,
// including calling tools if requested by the model.
func (a *Agent) ExecuteTask(prompt string) (string, error) {
	log.Printf("Agent '%s' executing task: %s", a.name, prompt)

	session := a.model.StartChat()
	resp, err := session.SendMessage(a.manager.ctx, genai.Text(prompt))
	if err != nil {
		return "", fmt.Errorf("failed to send message to Gemini: %w", err)
	}

	// Check if the model wants to call a tool
	if len(resp.Candidates) > 0 && len(resp.Candidates[0].Content.Parts) > 0 {
		part := resp.Candidates[0].Content.Parts[0]
		if fc, ok := part.(genai.FunctionCall); ok {
			// The model wants to call a function.
			log.Printf("Model wants to call tool: %s", fc.Name)

			tool, exists := a.toolFuncs[fc.Name]
			if !exists {
				return "", fmt.Errorf("tool '%s' not found", fc.Name)
			}

			// Convert args map to a JSON string for the tool function
			argsJSON, err := json.Marshal(fc.Args)
			if err != nil {
				return "", fmt.Errorf("failed to marshal tool args: %w", err)
			}

			// Execute the tool
			result, err := tool(string(argsJSON))
			if err != nil {
				return "", fmt.Errorf("tool '%s' execution failed: %w", fc.Name, err)
			}

			// Send the tool's result back to the model
			log.Printf("Sending tool result back to model: %s", result)
			resp, err = session.SendMessage(a.manager.ctx, genai.FunctionResponse{
				Name:     fc.Name,
				Response: map[string]any{"result": result},
			})
			if err != nil {
				return "", fmt.Errorf("failed to send tool response to Gemini: %w", err)
			}
		}
	}

	// Extract and return the final text response
	if len(resp.Candidates) > 0 && len(resp.Candidates[0].Content.Parts) > 0 {
		part := resp.Candidates[0].Content.Parts[0]
		if txt, ok := part.(genai.Text); ok {
			return string(txt), nil
		}
	}

	return "", fmt.Errorf("no text response from model")
}
