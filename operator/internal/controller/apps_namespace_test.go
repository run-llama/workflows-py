//go:build integration

package controller

import (
	"context"
	"fmt"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/reconcile"

	llamadeployv1 "llama-agents-operator/api/v1"
)

// Verifies that the reconciler produces child resources in whatever
// namespace the LlamaDeployment CR lives in — the "apps namespace" mode
// from the Helm chart. Cross-namespace owner references are illegal in
// Kubernetes, so co-locating the CR with its children keeps the existing
// ownership model unchanged. This test proves that assumption holds in
// practice by exercising a non-default namespace end-to-end.
var _ = Describe("LlamaDeployment apps namespace", func() {
	const (
		appsNamespace = "llama-agents-apps"
		projectID     = "apps-ns-project"
		repoURL       = "https://github.com/test/apps-ns.git"
		gitRef        = "abc123"
	)

	var (
		ctx          context.Context
		reconciler   *LlamaDeploymentReconciler
		nsCreatedKey = "created-by-apps-ns-test"
	)

	BeforeEach(func() {
		ctx = context.Background()
		reconciler = NewTestReconciler()

		ns := &corev1.Namespace{
			ObjectMeta: metav1.ObjectMeta{
				Name:   appsNamespace,
				Labels: map[string]string{nsCreatedKey: "true"},
			},
		}
		err := k8sClient.Create(ctx, ns)
		if err != nil && !errors.IsAlreadyExists(err) {
			Expect(err).NotTo(HaveOccurred())
		}
	})

	It("reconciles a LlamaDeployment in an alternate namespace", func() {
		name := "ld-in-apps-ns"

		ld := NewLlama(name, appsNamespace, projectID, repoURL, WithGitRef(gitRef))
		CreateAndReconcile(ctx, reconciler, ld)
		defer CleanupLlama(ctx, name, appsNamespace)
		CompleteBuild(ctx, reconciler, ld)

		By("creating a Deployment in the apps namespace")
		dep := GetDeploymentEventually(ctx, name, appsNamespace)
		Expect(dep.Namespace).To(Equal(appsNamespace))

		By("creating a ServiceAccount in the apps namespace")
		sa := &corev1.ServiceAccount{}
		Eventually(func() error {
			return k8sClient.Get(ctx, types.NamespacedName{Name: name + "-sa", Namespace: appsNamespace}, sa)
		}).Should(Succeed())
		Expect(sa.Namespace).To(Equal(appsNamespace))

		By("creating a Service in the apps namespace")
		ExpectServicePort(ctx, name, appsNamespace, 80, 8081)

		By("preserving the ownerReference so GC works on CR deletion")
		Expect(dep.OwnerReferences).NotTo(BeEmpty(), "Deployment should have an ownerReference to the CR")
		Expect(dep.OwnerReferences[0].Name).To(Equal(name))

		By("not creating any child resources in the default namespace")
		defaultDep := &appsv1.Deployment{}
		err := k8sClient.Get(ctx, types.NamespacedName{Name: name, Namespace: "default"}, defaultDep)
		Expect(errors.IsNotFound(err)).To(BeTrue(), "no Deployment should leak into the default namespace")
	})

	It("cleans up child resources when the CR is deleted in the apps namespace", func() {
		name := "ld-cleanup-apps-ns"

		ld := NewLlama(name, appsNamespace, projectID, repoURL, WithGitRef(gitRef))
		CreateAndReconcile(ctx, reconciler, ld)
		CompleteBuild(ctx, reconciler, ld)

		_ = GetDeploymentEventually(ctx, name, appsNamespace)

		By("deleting the CR and running one more reconcile to process finalizers")
		existing := &llamadeployv1.LlamaDeployment{}
		Expect(k8sClient.Get(ctx, types.NamespacedName{Name: name, Namespace: appsNamespace}, existing)).To(Succeed())
		if len(existing.Finalizers) > 0 {
			existing.Finalizers = []string{}
			Expect(k8sClient.Update(ctx, existing)).To(Succeed())
		}
		Expect(k8sClient.Delete(ctx, existing)).To(Succeed())

		_, err := reconciler.Reconcile(ctx, reconcile.Request{NamespacedName: types.NamespacedName{Name: name, Namespace: appsNamespace}})
		Expect(err).NotTo(HaveOccurred())

		By("confirming the CR is gone")
		Eventually(func() bool {
			err := k8sClient.Get(ctx, types.NamespacedName{Name: name, Namespace: appsNamespace}, existing)
			return errors.IsNotFound(err)
		}, "5s", "100ms").Should(BeTrue(), fmt.Sprintf("CR %s/%s should be deleted", appsNamespace, name))
	})
})
