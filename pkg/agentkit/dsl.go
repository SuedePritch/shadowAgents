package agentkit

// -------- Tool Builder --------
type ToolDefBuilder interface {
	ID(string) ToolDefBuilder
	Describe(string) ToolDefBuilder
	Register(app *App) ToolDef
}

type toolBuilder struct {
	id   string
	desc string
}

func NewToolBuilder(defaultID string) ToolDefBuilder {
	return &toolBuilder{id: defaultID}
}
func (t *toolBuilder) ID(id string) ToolDefBuilder        { t.id = id; return t }
func (t *toolBuilder) Describe(d string) ToolDefBuilder   { t.desc = d; return t }
func (t *toolBuilder) Register(app *App) ToolDef {
	td := ToolDef{ID: t.id, Description: t.desc}
	if app.tools == nil { app.tools = map[string]ToolDef{} }
	app.tools[td.ID] = td
	return td
}

// -------- Agent Builder --------
type AgentDefBuilder interface {
	System(string) AgentDefBuilder
	AddTools(ids ...string) AgentDefBuilder
	Register(app *App) AgentDef
}

type agentBuilder struct {
	id      string
	system  string
	toolIDs []string
}

func (a *App) InstantiateAgent(id string) AgentDefBuilder { return &agentBuilder{id: id} }

func (b *agentBuilder) System(s string) AgentDefBuilder    { b.system = s; return b }
func (b *agentBuilder) AddTools(ids ...string) AgentDefBuilder { b.toolIDs = append(b.toolIDs, ids...); return b }
func (b *agentBuilder) Register(app *App) AgentDef {
	ad := AgentDef{ID: b.id, System: b.system, ToolIDs: b.toolIDs}
	if app.agents == nil { app.agents = map[string]AgentDef{} }
	app.agents[ad.ID] = ad
	return ad
}

// -------- Supervisor Builder --------
type SupervisorDefBuilder interface {
	AddAgents(ids ...string) SupervisorDefBuilder
	AddTools(ids ...string) SupervisorDefBuilder
	AutoRoute(bool) SupervisorDefBuilder
	Register(app *App) *SupervisorDef
}

type supervisorBuilder struct {
	agentIDs []string
	toolIDs  []string
	auto     bool
}

func (a *App) Supervisor() SupervisorDefBuilder { return &supervisorBuilder{} }

func (b *supervisorBuilder) AddAgents(ids ...string) SupervisorDefBuilder { b.agentIDs = append(b.agentIDs, ids...); return b }
func (b *supervisorBuilder) AddTools(ids ...string) SupervisorDefBuilder  { b.toolIDs = append(b.toolIDs, ids...); return b }
func (b *supervisorBuilder) AutoRoute(v bool) SupervisorDefBuilder        { b.auto = v; return b }
func (b *supervisorBuilder) Register(app *App) *SupervisorDef {
	app.supervisor = &SupervisorDef{AgentIDs: b.agentIDs, ToolIDs: b.toolIDs, AutoRoute: b.auto}
	return app.supervisor
}

// Convenience: register a tool via a builder
func (a *App) CreateTool(b ToolDefBuilder) ToolDef { return b.Register(a) }
