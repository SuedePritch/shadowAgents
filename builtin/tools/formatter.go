package tools

import "agentframework/pkg/agentkit"

func Formatter() agentkit.ToolDefBuilder {
	return agentkit.NewToolBuilder("Formatter").Describe("Standardize and polish final output.")
}
