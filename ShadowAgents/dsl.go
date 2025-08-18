package shadowagents

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"reflect"
	"strings"

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

// registerToolInternal is the original tool registration logic.
func (a *Agent) registerToolInternal(name, description string, params *genai.Schema, toolFunc Tool) {
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

// generateSchemaFromStruct uses reflection to create a genai.Schema from a struct.
// It reads `json` and `description` tags from the struct fields.
func generateSchemaFromStruct(paramType reflect.Type) (*genai.Schema, error) {
	properties := make(map[string]*genai.Schema)
	var required []string

	for i := 0; i < paramType.NumField(); i++ {
		field := paramType.Field(i)
		jsonTag := field.Tag.Get("json")
		if jsonTag == "" || jsonTag == "-" {
			continue // Skip fields without a json tag
		}

		parts := strings.Split(jsonTag, ",")
		propName := parts[0]

		isOptional := false
		for _, part := range parts[1:] {
			if part == "omitempty" {
				isOptional = true
				break
			}
		}
		if !isOptional {
			required = append(required, propName)
		}

		desc := field.Tag.Get("description")

		var propType genai.Type
		switch field.Type.Kind() {
		case reflect.String:
			propType = genai.TypeString
		case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
			propType = genai.TypeInteger
		case reflect.Float32, reflect.Float64:
			propType = genai.TypeNumber
		case reflect.Bool:
			propType = genai.TypeBoolean
		default:
			return nil, fmt.Errorf("unsupported type for field %s: %s", field.Name, field.Type.Kind())
		}

		properties[propName] = &genai.Schema{
			Type:        propType,
			Description: desc,
		}
	}

	return &genai.Schema{
		Type:       genai.TypeObject,
		Properties: properties,
		Required:   required,
	}, nil
}

// RegisterTool simplifies tool creation by using reflection.
// It accepts a function with the signature `func(T) (string, error)` where T is a struct.
// The framework automatically generates the schema and handles JSON parsing.
func (a *Agent) RegisterTool(name, description string, toolFunc any) error {
	val := reflect.ValueOf(toolFunc)
	if val.Kind() != reflect.Func {
		return fmt.Errorf("toolFunc must be a function")
	}
	typ := val.Type()
	if typ.NumIn() != 1 || typ.NumOut() != 2 {
		return fmt.Errorf("toolFunc must have the signature func(T) (string, error)")
	}
	paramType := typ.In(0)
	if paramType.Kind() != reflect.Struct {
		return fmt.Errorf("toolFunc's first argument must be a struct")
	}
	if typ.Out(0).Kind() != reflect.String || typ.Out(1).Name() != "error" {
		return fmt.Errorf("toolFunc must return (string, error)")
	}

	schema, err := generateSchemaFromStruct(paramType)
	if err != nil {
		return fmt.Errorf("failed to generate schema from struct: %w", err)
	}

	wrapperFunc := func(args string) (string, error) {
		paramValuePtr := reflect.New(paramType)
		if err := json.Unmarshal([]byte(args), paramValuePtr.Interface()); err != nil {
			return "", fmt.Errorf("failed to unmarshal args into struct: %w", err)
		}

		results := val.Call([]reflect.Value{paramValuePtr.Elem()})

		resultStr := results[0].String()
		var resultErr error
		if !results[1].IsNil() {
			resultErr = results[1].Interface().(error)
		}

		return resultStr, resultErr
	}

	a.registerToolInternal(name, description, schema, wrapperFunc)
	return nil
}

// NewSubagent creates a new agent that is a child of the current agent.
// This is useful for creating hierarchical agent structures.
func (a *Agent) NewSubagent(name, systemPrompt string) *Agent {
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

	if len(resp.Candidates) > 0 && len(resp.Candidates[0].Content.Parts) > 0 {
		part := resp.Candidates[0].Content.Parts[0]
		if fc, ok := part.(genai.FunctionCall); ok {
			log.Printf("Model wants to call tool: %s", fc.Name)
			tool, exists := a.toolFuncs[fc.Name]
			if !exists {
				return "", fmt.Errorf("tool '%s' not found", fc.Name)
			}

			argsJSON, err := json.Marshal(fc.Args)
			if err != nil {
				return "", fmt.Errorf("failed to marshal tool args: %w", err)
			}

			result, err := tool(string(argsJSON))
			if err != nil {
				return "", fmt.Errorf("tool '%s' execution failed: %w", fc.Name, err)
			}

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

	if len(resp.Candidates) > 0 && len(resp.Candidates[0].Content.Parts) > 0 {
		part := resp.Candidates[0].Content.Parts[0]
		if txt, ok := part.(genai.Text); ok {
			return string(txt), nil
		}
	}

	return "", fmt.Errorf("no text response from model")
}