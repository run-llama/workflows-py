//go:build integration

package controller

import (
	"context"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	"github.com/prometheus/client_golang/prometheus"
	dto "github.com/prometheus/client_model/go"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"

	llamadeployv1 "llama-agents-operator/api/v1"
)

var _ = Describe("PhaseCollector", func() {
	const testNamespace = "default"

	It("should report counts by phase", func() {
		ctx := context.Background()

		// Create test LlamaDeployments with different phases.
		ld1 := &llamadeployv1.LlamaDeployment{
			ObjectMeta: metav1.ObjectMeta{Name: "test-metrics-phase-1", Namespace: testNamespace},
			Spec: llamadeployv1.LlamaDeploymentSpec{
				ProjectId: "metrics-test",
				RepoUrl:   "https://github.com/test/repo.git",
				GitRef:    "main",
			},
		}
		ld2 := &llamadeployv1.LlamaDeployment{
			ObjectMeta: metav1.ObjectMeta{Name: "test-metrics-phase-2", Namespace: testNamespace},
			Spec: llamadeployv1.LlamaDeploymentSpec{
				ProjectId: "metrics-test",
				RepoUrl:   "https://github.com/test/repo.git",
				GitRef:    "main",
				Suspended: true,
			},
		}

		for _, ld := range []*llamadeployv1.LlamaDeployment{ld1, ld2} {
			Expect(k8sClient.Create(ctx, ld)).To(Succeed())
		}
		defer func() {
			for _, ld := range []*llamadeployv1.LlamaDeployment{ld1, ld2} {
				_ = k8sClient.Delete(ctx, ld)
			}
		}()

		// Set phases via status subresource
		Expect(k8sClient.Get(ctx, types.NamespacedName{Name: ld1.Name, Namespace: ld1.Namespace}, ld1)).To(Succeed())
		ld1.Status.Phase = PhaseRunning
		Expect(k8sClient.Status().Update(ctx, ld1)).To(Succeed())

		Expect(k8sClient.Get(ctx, types.NamespacedName{Name: ld2.Name, Namespace: ld2.Namespace}, ld2)).To(Succeed())
		ld2.Status.Phase = PhaseSuspended
		Expect(k8sClient.Status().Update(ctx, ld2)).To(Succeed())

		collector := NewPhaseCollector(k8sClient)

		// Collect metrics
		ch := make(chan prometheus.Metric, 100)
		collector.Collect(ch)
		close(ch)

		counts := make(map[string]float64)
		for m := range ch {
			d := &dto.Metric{}
			Expect(m.Write(d)).To(Succeed())
			phase := d.Label[0].GetValue()
			counts[phase] += d.Gauge.GetValue()
		}

		Expect(counts[PhaseRunning]).To(BeNumerically(">=", 1))
		Expect(counts[PhaseSuspended]).To(BeNumerically(">=", 1))
	})
})
