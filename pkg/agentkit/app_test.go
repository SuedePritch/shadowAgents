package agentkit_test

import (
	"testing"
	"agentframework/pkg/agentkit"
	"agentframework/builtin/tools"
	"agentframework/pkg/agentkit/adapters/vertex"
)

func TestAgentFrameworkMVP(t *testing.T) {
	t.Log("Starting AgentFramework MVP test...")
	app := agentkit.NewApp().
		WithProvider(vertex.Vertex().Project("demo-project").Location("us-central1").Model("gemini-2.5-pro").Build()).
		Boot()

	t.Log("Registering tools...")
	app.CreateTool(tools.Math())
	app.CreateTool(tools.TodoVerifier())
	app.CreateTool(tools.Formatter())

	t.Log("Registering agent...")
	app.InstantiateAgent("Writer").
		System("You write concise, structured briefs.").
		AddTools("Math", "Formatter", "TodoVerifier").
		Register(app)

	t.Log("Registering supervisor...")
	app.Supervisor().
		AddAgents("Writer").
		AddTools("Math").
		AutoRoute(true).
		Register(app)

	t.Log("Running supervisor instance...")
	output := app.SupervisorInstance().
		WithContext("audience=exec, locale=en-CA").
		ExpectJSON("Brief@v1").
		Run("Draft a one-pager for Project Atlas including a simple budget calc.")

	t.Logf("SupervisorInstance.Run output: %s", output)
	if output == "" {
		t.Errorf("Expected non-empty output from SupervisorInstance.Run, got empty string")
	}
}
