package shadowagents
import (
	"context"
	"fmt"
	"os"
	"strings"
	"testing"
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

	// --- 4. Register a Tool and Execute a Task (Simplified) ---
	t.Run("RegisterToolAndExecuteTask", func(t *testing.T) {
		taskAgent := manager.NewAgent(
			"Tasker",
			"You are an agent that finds user emails.",
		)

		// Define the struct for the tool's parameters
		type getEmailParams struct {
			Username string `json:"username" description:"The username to look up."`
		}

		// Register a mock tool using the simplified method
		const expectedEmail = "test.user@example.com"
		getEmailFunc := func(p getEmailParams) (string, error) {
			if p.Username == "testuser" {
				// The result should be a simple string, not JSON.
				// The framework and model handle the JSON structure.
				return fmt.Sprintf("The email for testuser is %s.", expectedEmail), nil
			}
			return "Email not found.", nil
		}

		err := taskAgent.RegisterTool(
			"get_email_by_username",
			"Gets a user's email address by their username.",
			getEmailFunc,
		)
		if err != nil {
			t.Fatalf("RegisterTool() failed: %v", err)
		}

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