package agentkit

import (
	"fmt"
	"strings"
	"time"
)

type App struct {
	provider    Provider
	tools       map[string]ToolDef
	agents      map[string]AgentDef
	supervisor  *SupervisorDef
}

func NewApp() *App {
	return &App{
		tools:  make(map[string]ToolDef),
		agents: make(map[string]AgentDef),
	}
}

type Provider interface{ Name() string }

type ToolDef struct {
	ID          string
	Description string
}

type AgentDef struct {
	ID       string
	System   string
	ToolIDs  []string
}

type SupervisorDef struct {
	AgentIDs []string
	ToolIDs  []string
	AutoRoute bool
}

type Supervisor interface {
	WithContext(kv string) Supervisor
	ExpectJSON(schemaID string) Supervisor
	Run(goal string) string
}

type supervisorInstance struct {
	app       *App
	contextKV string
	expectSchema string
}

func (s *supervisorInstance) WithContext(kv string) Supervisor { s.contextKV = kv; return s }
func (s *supervisorInstance) ExpectJSON(schemaID string) Supervisor { s.expectSchema = schemaID; return s }
func (s *supervisorInstance) Run(goal string) string {
	agents := []string{}
	tools := []string{}
	if s.app.supervisor != nil {
		agents = append(agents, s.app.supervisor.AgentIDs...)
		tools = append(tools, s.app.supervisor.ToolIDs...)
	}
	// MVP stub output so you can see wiring works
	return fmt.Sprintf("[MVP] %s | ctx=%q | expect=%q | agents=%s | tools=%s | time=%s",
		goal, s.contextKV, s.expectSchema,
		strings.Join(agents, ","), strings.Join(tools, ","),
		time.Now().Format(time.RFC3339))
}
