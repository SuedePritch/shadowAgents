package tools

import "agentframework/pkg/agentkit"

func Math() agentkit.ToolDefBuilder {
	return agentkit.NewToolBuilder("Math").Describe("Basic arithmetic on two numbers.")
}
