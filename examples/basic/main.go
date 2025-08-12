package main

import (
	"fmt"

	"agentframework/builtin/tools"
	af "agentframework/pkg/agentkit"
	vx "agentframework/pkg/agentkit/adapters/vertex"
)

func main() {
	app := af.NewApp().
		WithProvider(vx.Vertex().Project("demo-project").Location("us-central1").Model("gemini-2.5-pro").Build()).
		Boot()

	// Tools
	app.CreateTool(tools.Math())
	app.CreateTool(tools.TodoVerifier())
	app.CreateTool(tools.Formatter())

	// Agents
	app.InstantiateAgent("Writer").
		System("You write concise, structured briefs.").
		AddTools("Math", "Formatter", "TodoVerifier").
		Register(app)

	// Supervisor
	app.Supervisor().
		AddAgents("Writer").
		AddTools("Math").
		AutoRoute(true).
		Register(app)

	// Run
	ans := app.SupervisorInstance().
		WithContext("audience=exec, locale=en-CA").
		ExpectJSON("Brief@v1").
		Run("Draft a one-pager for Project Atlas including a simple budget calc.")

	fmt.Println(ans)
}
