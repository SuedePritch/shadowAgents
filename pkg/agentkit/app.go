package agentkit

// App lifecycle surface (MVP stubs)

func (a *App) WithProvider(p Provider) *App { a.provider = p; return a }
func (a *App) Boot() *App { return a } // placeholder for validation/warmup

// Instance handles
func (a *App) SupervisorInstance() Supervisor {
	return &supervisorInstance{app: a}
}
