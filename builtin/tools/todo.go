package tools

import "agentframework/pkg/agentkit"

func TodoVerifier() agentkit.ToolDefBuilder {
	return agentkit.NewToolBuilder("TodoVerifier").Describe("Check that all required steps are complete and list missing ones.")
}
