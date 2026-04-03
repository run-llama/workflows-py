package controller

import (
	"context"
	"fmt"

	. "github.com/onsi/gomega"
	appsv1 "k8s.io/api/apps/v1"
	batchv1 "k8s.io/api/batch/v1"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/tools/record"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/reconcile"

	llamadeployv1 "llama-agents-operator/api/v1"
)

// GetDeploymentEventually fetches a Deployment with Eventually retry.
func GetDeploymentEventually(ctx context.Context, name, ns string) *appsv1.Deployment {
	depl := &appsv1.Deployment{}
	Eventually(func() error {
		return k8sClient.Get(ctx, types.NamespacedName{Name: name, Namespace: ns}, depl)
	}).Should(Succeed())
	return depl
}

// GetConfigMapEventually fetches a ConfigMap with Eventually retry.
func GetConfigMapEventually(ctx context.Context, name, ns string) *corev1.ConfigMap {
	cm := &corev1.ConfigMap{}
	Eventually(func() error {
		return k8sClient.Get(ctx, types.NamespacedName{Name: name, Namespace: ns}, cm)
	}).Should(Succeed())
	return cm
}

// FindContainer returns a container by name from a PodSpec.
func FindContainer(podSpec corev1.PodSpec, name string) (corev1.Container, bool) {
	for _, c := range podSpec.Containers {
		if c.Name == name {
			return c, true
		}
	}
	return corev1.Container{}, false
}

// FindInitContainer returns an init container by name from a PodSpec.
func FindInitContainer(podSpec corev1.PodSpec, name string) (corev1.Container, bool) {
	for _, c := range podSpec.InitContainers {
		if c.Name == name {
			return c, true
		}
	}
	return corev1.Container{}, false
}

// EnvMap builds a map of non-empty env var values from a container.
func EnvMap(c corev1.Container) map[string]string {
	out := make(map[string]string, len(c.Env))
	for _, e := range c.Env {
		if e.Value != "" {
			out[e.Name] = e.Value
		}
	}
	return out
}

// ExpectEnv asserts an exact env var value on a container.
func ExpectEnv(c corev1.Container, key, value string) {
	envs := EnvMap(c)
	Expect(envs[key]).To(Equal(value), fmt.Sprintf("env %s mismatch", key))
}

// ExpectEnvMatches asserts an env var matches a regex pattern.
func ExpectEnvMatches(c corev1.Container, key, pattern string) {
	envs := EnvMap(c)
	Expect(envs[key]).To(MatchRegexp(pattern), fmt.Sprintf("env %s mismatch", key))
}

// ExpectServicePort asserts a service exposes port->target mapping.
func ExpectServicePort(ctx context.Context, name, ns string, port int32, target int) {
	svc := &corev1.Service{}
	Eventually(func() error {
		return k8sClient.Get(ctx, types.NamespacedName{Name: name, Namespace: ns}, svc)
	}).Should(Succeed())
	Expect(svc.Spec.Ports).To(HaveLen(1))
	Expect(svc.Spec.Ports[0].Port).To(Equal(port))
	Expect(svc.Spec.Ports[0].TargetPort.IntValue()).To(Equal(target))
}

// CleanupLlama deletes a LlamaDeployment and associated ServiceAccount if present.
func CleanupLlama(ctx context.Context, name, ns string) {
	// Delete LlamaDeployment with finalizer handling
	llamaDeploy := &llamadeployv1.LlamaDeployment{}
	if err := k8sClient.Get(ctx, types.NamespacedName{Name: name, Namespace: ns}, llamaDeploy); err == nil {
		if len(llamaDeploy.Finalizers) > 0 {
			llamaDeploy.Finalizers = []string{}
			_ = k8sClient.Update(ctx, llamaDeploy)
		}
		_ = k8sClient.Delete(ctx, llamaDeploy)
		// Wait until deleted
		Eventually(func() bool {
			err := k8sClient.Get(ctx, types.NamespacedName{Name: name, Namespace: ns}, llamaDeploy)
			return apierrors.IsNotFound(err)
		}, "5s", "100ms").Should(BeTrue())
	}

	// Delete ServiceAccount if exists
	sa := &corev1.ServiceAccount{}
	if err := k8sClient.Get(ctx, types.NamespacedName{Name: name + "-sa", Namespace: ns}, sa); err == nil {
		_ = k8sClient.Delete(ctx, sa)
	}
}

// Functional options for building LlamaDeployment spec
type LlamaSpecOption func(*llamadeployv1.LlamaDeploymentSpec)

func WithImage(image string) LlamaSpecOption {
	return func(s *llamadeployv1.LlamaDeploymentSpec) { s.Image = image }
}
func WithImageTag(tag string) LlamaSpecOption {
	return func(s *llamadeployv1.LlamaDeploymentSpec) { s.ImageTag = tag }
}
func WithSecret(name string) LlamaSpecOption {
	return func(s *llamadeployv1.LlamaDeploymentSpec) { s.SecretName = name }
}
func WithDeploymentFilePath(path string) LlamaSpecOption {
	return func(s *llamadeployv1.LlamaDeploymentSpec) { s.DeploymentFilePath = path }
}
func WithGitRef(ref string) LlamaSpecOption {
	return func(s *llamadeployv1.LlamaDeploymentSpec) { s.GitRef = ref }
}
func WithGitSha(sha string) LlamaSpecOption {
	return func(s *llamadeployv1.LlamaDeploymentSpec) { s.GitSha = sha }
}
func WithRepoUrl(url string) LlamaSpecOption {
	return func(s *llamadeployv1.LlamaDeploymentSpec) { s.RepoUrl = url }
}
func WithAssetsPath(path string) LlamaSpecOption {
	return func(s *llamadeployv1.LlamaDeploymentSpec) { s.StaticAssetsPath = path }
}
func WithBuildGeneration(gen int64) LlamaSpecOption {
	return func(s *llamadeployv1.LlamaDeploymentSpec) { s.BuildGeneration = gen }
}

// NewLlama creates a LlamaDeployment with defaults and applies options.
func NewLlama(name, ns, projectId, repoURL string, opts ...LlamaSpecOption) *llamadeployv1.LlamaDeployment {
	spec := llamadeployv1.LlamaDeploymentSpec{
		ProjectId: projectId,
		RepoUrl:   repoURL,
	}
	for _, opt := range opts {
		opt(&spec)
	}
	return &llamadeployv1.LlamaDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: ns},
		Spec:       spec,
	}
}

// SetDeploymentAvailableReplicas sets AvailableReplicas on a Deployment's status.
// Use this in tests that need checkRolloutTimeout to pick PhaseRolloutFailed (>0)
// vs PhaseFailed (0), since envtest defaults to 0 available replicas.
func SetDeploymentAvailableReplicas(ctx context.Context, name, ns string, replicas int32) {
	deployment := &appsv1.Deployment{}
	Expect(k8sClient.Get(ctx, types.NamespacedName{Name: name, Namespace: ns}, deployment)).To(Succeed())
	deployment.Status.AvailableReplicas = replicas
	if replicas > 0 {
		deployment.Status.ReadyReplicas = replicas
		deployment.Status.Replicas = replicas
	}
	Expect(k8sClient.Status().Update(ctx, deployment)).To(Succeed())
}

// ReconcilerOption configures a LlamaDeploymentReconciler created by NewTestReconciler.
type ReconcilerOption func(*LlamaDeploymentReconciler)

func WithMaxConcurrentRollouts(n int) ReconcilerOption {
	return func(r *LlamaDeploymentReconciler) { r.MaxConcurrentRollouts = n }
}

func WithMaxDeployments(n int) ReconcilerOption {
	return func(r *LlamaDeploymentReconciler) { r.MaxDeployments = n }
}

// NewTestReconciler creates a reconciler wired to the test envtest client.
func NewTestReconciler(opts ...ReconcilerOption) *LlamaDeploymentReconciler {
	r := &LlamaDeploymentReconciler{
		Client:       k8sClient,
		Scheme:       k8sClient.Scheme(),
		Recorder:     record.NewFakeRecorder(100),
		DirectClient: k8sClient,
	}
	for _, o := range opts {
		o(r)
	}
	return r
}

// CreateAndReconcile creates the CR and runs one reconciliation.
func CreateAndReconcile(ctx context.Context, r *LlamaDeploymentReconciler, obj *llamadeployv1.LlamaDeployment) {
	Expect(k8sClient.Create(ctx, obj)).To(Succeed())
	_, err := r.Reconcile(ctx, reconcile.Request{NamespacedName: types.NamespacedName{Name: obj.Name, Namespace: obj.Namespace}})
	Expect(err).NotTo(HaveOccurred())
}

// CompleteBuild marks the build Job for a LlamaDeployment as succeeded and reconciles
// again so that reconcileBuild proceeds past the build phase and creates the Deployment.
// Call this after the first Reconcile when spec.GitSha is set.
func CompleteBuild(ctx context.Context, r *LlamaDeploymentReconciler, llamaDeploy *llamadeployv1.LlamaDeployment) {
	// Re-read to get latest status (buildId set by first reconcile)
	Expect(k8sClient.Get(ctx, types.NamespacedName{Name: llamaDeploy.Name, Namespace: llamaDeploy.Namespace}, llamaDeploy)).To(Succeed())
	buildId := computeBuildId(llamaDeploy)
	jobName := fmt.Sprintf("%s-build-%s", llamaDeploy.Name, buildId)
	if len(jobName) > 63 {
		jobName = jobName[:63]
	}

	// Mark the build Job as succeeded
	job := &batchv1.Job{}
	Expect(k8sClient.Get(ctx, client.ObjectKey{Name: jobName, Namespace: llamaDeploy.Namespace}, job)).To(Succeed())
	job.Status.Succeeded = 1
	Expect(k8sClient.Status().Update(ctx, job)).To(Succeed())

	// Reconcile again to proceed past build
	_, err := r.Reconcile(ctx, reconcile.Request{NamespacedName: types.NamespacedName{Name: llamaDeploy.Name, Namespace: llamaDeploy.Namespace}})
	Expect(err).NotTo(HaveOccurred())
}
