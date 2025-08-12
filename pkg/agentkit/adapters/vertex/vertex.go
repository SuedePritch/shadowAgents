package vertex

type Provider struct {
	project     string
	location    string
	model       string
	temperature float32
}

func Vertex() *Provider { return &Provider{} }
func (p *Provider) Project(id string) *Provider     { p.project = id; return p }
func (p *Provider) Location(l string) *Provider     { p.location = l; return p }
func (p *Provider) Model(m string) *Provider        { p.model = m; return p }
func (p *Provider) Temperature(t float32) *Provider { p.temperature = t; return p }
func (p *Provider) Build() *Provider                { return p }
func (p *Provider) Name() string                    { return "vertex:" + p.model }
