package controller

import (
	"context"

	"github.com/prometheus/client_golang/prometheus"
	"sigs.k8s.io/controller-runtime/pkg/client"

	llamadeployv1 "llama-agents-operator/api/v1"
)

var descPhaseTotal = prometheus.NewDesc(
	"llamadeployments_by_phase",
	"Number of LlamaDeployments per status phase",
	[]string{"phase"}, nil,
)

// PhaseCollector implements prometheus.Collector and reports deployment counts
// by phase, reading from the informer cache at scrape time.
type PhaseCollector struct {
	reader client.Reader
}

// NewPhaseCollector creates a new collector that uses the given reader
// (typically the controller-runtime manager's cached client) to list LlamaDeployments.
func NewPhaseCollector(reader client.Reader) *PhaseCollector {
	return &PhaseCollector{reader: reader}
}

func (c *PhaseCollector) Describe(ch chan<- *prometheus.Desc) {
	ch <- descPhaseTotal
}

func (c *PhaseCollector) Collect(ch chan<- prometheus.Metric) {
	var list llamadeployv1.LlamaDeploymentList
	if err := c.reader.List(context.Background(), &list); err != nil {
		return
	}

	counts := make(map[string]float64)
	for i := range list.Items {
		phase := list.Items[i].Status.Phase
		if phase == "" {
			phase = "Unknown"
		}
		counts[phase]++
	}

	for phase, count := range counts {
		ch <- prometheus.MustNewConstMetric(descPhaseTotal, prometheus.GaugeValue, count, phase)
	}
}

var descActiveTotal = prometheus.NewDesc(
	"llamadeployments_active_total",
	"Number of active (non-suspended, non-failed) LlamaDeployments",
	[]string{"namespace"}, nil,
)

var descMaxCapacity = prometheus.NewDesc(
	"llamadeployments_max_capacity",
	"Configured maximum deployments limit (0 = unlimited)",
	[]string{"namespace"}, nil,
)

// CapacityCollector implements prometheus.Collector and reports active deployment
// counts and the configured maximum capacity per namespace.
type CapacityCollector struct {
	reader         client.Reader
	maxDeployments int
}

// NewCapacityCollector creates a new collector that reports active deployment
// counts and the configured max deployments limit.
func NewCapacityCollector(reader client.Reader, maxDeployments int) *CapacityCollector {
	return &CapacityCollector{reader: reader, maxDeployments: maxDeployments}
}

func (c *CapacityCollector) Describe(ch chan<- *prometheus.Desc) {
	ch <- descActiveTotal
	ch <- descMaxCapacity
}

func (c *CapacityCollector) Collect(ch chan<- prometheus.Metric) {
	var list llamadeployv1.LlamaDeploymentList
	if err := c.reader.List(context.Background(), &list); err != nil {
		return
	}

	counts := make(map[string]float64)
	namespaces := make(map[string]bool)
	for i := range list.Items {
		ns := list.Items[i].Namespace
		namespaces[ns] = true
		phase := list.Items[i].Status.Phase
		if isActivePhase(phase) {
			counts[ns]++
		}
	}

	for ns := range namespaces {
		ch <- prometheus.MustNewConstMetric(descActiveTotal, prometheus.GaugeValue, counts[ns], ns)
		ch <- prometheus.MustNewConstMetric(descMaxCapacity, prometheus.GaugeValue, float64(c.maxDeployments), ns)
	}
}
