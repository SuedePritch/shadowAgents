package shadowagents

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"testing"

	"github.com/google/generative-ai-go/genai"
)

// TestAgentFramework runs an end-to-end test of the agent framework.
// It requires the GEMINI_API_KEY environment variable to be set.
func TestAgentFramework(t *testing.T) {
	// --- 1. Check for API Key ---
	apiKey := os.Getenv("GEMINI_API_KEY")
	if apiKey == "" {
		t.Skip("Skipping test: GEMINI_API_KEY environment variable not set.")
	}

	// --- 2. Setup ---
	ctx := context.Background()
	manager, err := Setup(ctx, apiKey)
	if err != nil {
		t.Fatalf("Setup() failed: %v", err)
	}
	defer manager.Close()

	// --- 3. Create a New Agent ---
	t.Run("NewAgent", func(t *testing.T) {
		testAgent := manager.NewAgent(
			"TestAgent",
			"You are a testing agent. Use tools when necessary.",
		)
		if testAgent == nil {
			t.Fatal("NewAgent() returned nil")
		}
		if testAgent.name != "TestAgent" {
			t.Errorf("Expected agent name 'TestAgent', got '%s'", testAgent.name)
		}
		if manager.agents["TestAgent"] == nil {
			t.Error("Agent was not registered in the manager")
		}
	})

	// --- 4. Register a Tool and Execute a Task ---
	t.Run("RegisterToolAndExecuteTask", func(t *testing.T) {
		// Create an agent specifically for this test case
		taskAgent := manager.NewAgent(
			"Tasker",
			"You are an agent that finds user emails.",
		)

		// Register a mock tool
		const expectedEmail = "test.user@example.com"
		taskAgent.RegisterTool(
			"get_email_by_username",
			"Gets a user's email address by their username.",
			&genai.Schema{
				Type: genai.TypeObject,
				Properties: map[string]*genai.Schema{
					"username": {Type: genai.TypeString, Description: "The username to look up."},
				},
				Required: []string{"username"},
			},
			func(args string) (string, error) {
				var params struct {
					Username string `json:"username"`
				}
				if err := json.Unmarshal([]byte(args), &params); err != nil {
					return "", fmt.Errorf("failed to unmarshal args: %w", err)
				}
				if params.Username == "testuser" {
					return fmt.Sprintf(`{"email": "%s"}`, expectedEmail), nil
				}
				return `{"email": "not.found"}`, nil
			},
		)

		// Execute a task that should trigger the tool
		prompt := "What is the email for the user 'testuser'?"
		finalResponse, err := taskAgent.ExecuteTask(prompt)
		if err != nil {
			t.Fatalf("ExecuteTask() failed: %v", err)
		}

		// Assert that the final response contains the expected email
		if !strings.Contains(finalResponse, expectedEmail) {
			t.Errorf("Expected final response to contain '%s', but got: '%s'", expectedEmail, finalResponse)
		}
	})

	// --- 5. Create a Sub-agent ---
	t.Run("NewSubagent", func(t *testing.T) {
		parentAgent := manager.NewAgent("Parent", "I am a parent agent.")
		subAgent := parentAgent.NewSubagent("Child", "I am a child agent.")

		if subAgent == nil {
			t.Fatal("NewSubagent() returned nil")
		}
		if parentAgent.subAgents["Child"] == nil {
			t.Error("Sub-agent was not registered under the parent agent")
		}
		if manager.agents["Child"] == nil {
			t.Error("Sub-agent was not registered in the central manager")
		}
	})
}