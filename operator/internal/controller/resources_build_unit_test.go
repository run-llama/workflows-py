//go:build !integration

package controller

import (
	"context"
	"fmt"
	"strings"
	"testing"

	batchv1 "k8s.io/api/batch/v1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"

	llamadeployv1 "llama-agents-operator/api/v1"
)

func TestCreateBuildJob_HasDefaultResources(t *testing.T) {
	r := &LlamaDeploymentReconciler{}
	ld := &llamadeployv1.LlamaDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "my-app", Namespace: "default"},
		Spec: llamadeployv1.LlamaDeploymentSpec{
			ProjectId: "proj-123",
			RepoUrl:   "https://github.com/example/repo",
			GitRef:    "main",
		},
	}

	job := r.createBuildJob(ld, "abc123")

	containers := job.Spec.Template.Spec.Containers
	if len(containers) != 1 {
		t.Fatalf("expected 1 container, got %d", len(containers))
	}

	res := containers[0].Resources

	// Verify requests are set
	cpuReq := res.Requests[corev1.ResourceCPU]
	if cpuReq.Cmp(resource.MustParse("750m")) != 0 {
		t.Errorf("expected CPU request 750m, got %s", cpuReq.String())
	}
	memReq := res.Requests[corev1.ResourceMemory]
	if memReq.Cmp(resource.MustParse("2Gi")) != 0 {
		t.Errorf("expected memory request 2Gi, got %s", memReq.String())
	}

	// Verify limits are set
	memLimit := res.Limits[corev1.ResourceMemory]
	if memLimit.Cmp(resource.MustParse("4096Mi")) != 0 {
		t.Errorf("expected memory limit 4096Mi, got %s", memLimit.String())
	}
}

func TestApplyBuildJobTemplateOverlay_MergesAppResources(t *testing.T) {
	scheme := newTestScheme()

	// Template with only ephemeral-storage on the "app" container (no "build" container).
	// This should be merged into the build job's defaults, not replace them.
	tmpl := &llamadeployv1.LlamaDeploymentTemplate{
		ObjectMeta: metav1.ObjectMeta{Name: "default", Namespace: "default"},
		Spec: llamadeployv1.LlamaDeploymentTemplateSpec{
			PodSpec: corev1.PodTemplateSpec{
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{
						{
							Name: "app",
							Resources: corev1.ResourceRequirements{
								Requests: corev1.ResourceList{
									corev1.ResourceEphemeralStorage: resource.MustParse("1500Mi"),
								},
								Limits: corev1.ResourceList{
									corev1.ResourceEphemeralStorage: resource.MustParse("3Gi"),
								},
							},
						},
					},
				},
			},
		},
	}

	fakeClient := fake.NewClientBuilder().
		WithScheme(scheme).
		WithObjects(tmpl).
		Build()

	r := &LlamaDeploymentReconciler{Client: fakeClient, Scheme: scheme}

	ld := &llamadeployv1.LlamaDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "my-app", Namespace: "default"},
		Spec:       llamadeployv1.LlamaDeploymentSpec{TemplateName: "default"},
	}

	job := r.createBuildJob(ld, "abc123")

	ctx := context.Background()
	if err := r.applyBuildJobTemplateOverlay(ctx, ld, job); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	res := job.Spec.Template.Spec.Containers[0].Resources

	// Default CPU/memory requests must still be present
	cpuReq := res.Requests[corev1.ResourceCPU]
	if cpuReq.Cmp(resource.MustParse("750m")) != 0 {
		t.Errorf("expected CPU request 750m, got %s", cpuReq.String())
	}
	memReq := res.Requests[corev1.ResourceMemory]
	if memReq.Cmp(resource.MustParse("2Gi")) != 0 {
		t.Errorf("expected memory request 2Gi, got %s", memReq.String())
	}

	// ephemeral-storage from template must be merged in
	ephReq := res.Requests[corev1.ResourceEphemeralStorage]
	if ephReq.Cmp(resource.MustParse("1500Mi")) != 0 {
		t.Errorf("expected ephemeral-storage request 1500Mi, got %s", ephReq.String())
	}

	// Default memory limit must still be present
	memLimit := res.Limits[corev1.ResourceMemory]
	if memLimit.Cmp(resource.MustParse("4096Mi")) != 0 {
		t.Errorf("expected memory limit 4096Mi, got %s", memLimit.String())
	}

	// ephemeral-storage limit from template must be merged in
	ephLimit := res.Limits[corev1.ResourceEphemeralStorage]
	if ephLimit.Cmp(resource.MustParse("3Gi")) != 0 {
		t.Errorf("expected ephemeral-storage limit 3Gi, got %s", ephLimit.String())
	}
}

func TestApplyBuildJobTemplateOverlay_TemplateOverridesDefaults(t *testing.T) {
	scheme := newTestScheme()

	// Template with a "build" container that overrides CPU request and memory limit.
	tmpl := &llamadeployv1.LlamaDeploymentTemplate{
		ObjectMeta: metav1.ObjectMeta{Name: "default", Namespace: "default"},
		Spec: llamadeployv1.LlamaDeploymentTemplateSpec{
			PodSpec: corev1.PodTemplateSpec{
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{
						{
							Name: "build",
							Resources: corev1.ResourceRequirements{
								Requests: corev1.ResourceList{
									corev1.ResourceCPU: resource.MustParse("2"),
								},
								Limits: corev1.ResourceList{
									corev1.ResourceMemory: resource.MustParse("8Gi"),
								},
							},
						},
					},
				},
			},
		},
	}

	fakeClient := fake.NewClientBuilder().
		WithScheme(scheme).
		WithObjects(tmpl).
		Build()

	r := &LlamaDeploymentReconciler{Client: fakeClient, Scheme: scheme}

	ld := &llamadeployv1.LlamaDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "my-app", Namespace: "default"},
		Spec:       llamadeployv1.LlamaDeploymentSpec{TemplateName: "default"},
	}

	job := r.createBuildJob(ld, "abc123")

	ctx := context.Background()
	if err := r.applyBuildJobTemplateOverlay(ctx, ld, job); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	res := job.Spec.Template.Spec.Containers[0].Resources

	// Template CPU request should override the default 750m
	cpuReq := res.Requests[corev1.ResourceCPU]
	if cpuReq.Cmp(resource.MustParse("2")) != 0 {
		t.Errorf("expected CPU request 2, got %s", cpuReq.String())
	}

	// Default memory request should be backfilled
	memReq := res.Requests[corev1.ResourceMemory]
	if memReq.Cmp(resource.MustParse("2Gi")) != 0 {
		t.Errorf("expected memory request 2Gi (default), got %s", memReq.String())
	}

	// Template memory limit should override the default 4096Mi
	memLimit := res.Limits[corev1.ResourceMemory]
	if memLimit.Cmp(resource.MustParse("8Gi")) != 0 {
		t.Errorf("expected memory limit 8Gi, got %s", memLimit.String())
	}
}

// newTestScheme returns a scheme with all types needed for reconcileBuild tests.
func newTestScheme() *runtime.Scheme {
	scheme := runtime.NewScheme()
	_ = llamadeployv1.AddToScheme(scheme)
	_ = batchv1.AddToScheme(scheme)
	_ = corev1.AddToScheme(scheme)
	return scheme
}

// buildSupersedeFixture describes a "stale build vs new spec" scenario. If
// staleJobStatus is nil the stale Job is omitted from the fake client (models
// the case where the Job was already TTL-reaped).
type buildSupersedeFixture struct {
	deploymentName string
	staleBuildId   string
	newGitSha      string
	staleJobStatus *batchv1.JobStatus
}

// newBuildSupersedeFixture wires up an LlamaDeployment whose Status points at
// staleBuildId/Running and whose Spec forces a new buildId via newGitSha,
// optionally with a stale Job already in the cluster.
func newBuildSupersedeFixture(
	t *testing.T,
	f buildSupersedeFixture,
) (*LlamaDeploymentReconciler, *llamadeployv1.LlamaDeployment, client.Client) {
	t.Helper()
	scheme := newTestScheme()

	llamaDeploy := &llamadeployv1.LlamaDeployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:       f.deploymentName,
			Namespace:  "default",
			Generation: 2,
		},
		Spec: llamadeployv1.LlamaDeploymentSpec{
			ProjectId: "proj-123",
			RepoUrl:   "https://github.com/example/repo",
			GitRef:    "main",
			GitSha:    f.newGitSha,
		},
		Status: llamadeployv1.LlamaDeploymentStatus{
			BuildId:     f.staleBuildId,
			BuildStatus: BuildStatusRunning,
			Phase:       PhaseBuilding,
		},
	}

	objs := []client.Object{llamaDeploy}
	if f.staleJobStatus != nil {
		objs = append(objs, &batchv1.Job{
			ObjectMeta: metav1.ObjectMeta{
				Name:      buildJobName(f.deploymentName, f.staleBuildId),
				Namespace: "default",
				Labels: map[string]string{
					"deploy.llamaindex.ai/deployment": f.deploymentName,
					"deploy.llamaindex.ai/build-id":   f.staleBuildId,
				},
			},
			Status: *f.staleJobStatus,
		})
	}

	fakeClient := fake.NewClientBuilder().
		WithScheme(scheme).
		WithObjects(objs...).
		WithStatusSubresource(llamaDeploy).
		Build()

	return &LlamaDeploymentReconciler{Client: fakeClient, Scheme: scheme}, llamaDeploy, fakeClient
}

// ---------------------------------------------------------------------------
// computeBuildId tests
// ---------------------------------------------------------------------------

func TestComputeBuildId_Deterministic(t *testing.T) {
	ld := &llamadeployv1.LlamaDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "my-app"},
		Spec:       llamadeployv1.LlamaDeploymentSpec{GitSha: "abc123"},
	}
	id1 := computeBuildId(ld)
	id2 := computeBuildId(ld)
	if id1 != id2 {
		t.Errorf("expected deterministic result, got %q and %q", id1, id2)
	}
	if len(id1) != 16 {
		t.Errorf("expected 16-char hash, got %d chars: %q", len(id1), id1)
	}
}

func TestComputeBuildId_DifferentInputs(t *testing.T) {
	ld1 := &llamadeployv1.LlamaDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "app-a"},
		Spec:       llamadeployv1.LlamaDeploymentSpec{GitSha: "sha1"},
	}
	ld2 := &llamadeployv1.LlamaDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "app-b"},
		Spec:       llamadeployv1.LlamaDeploymentSpec{GitSha: "sha1"},
	}
	if computeBuildId(ld1) == computeBuildId(ld2) {
		t.Error("expected different build IDs for different deployment names")
	}
}

func TestComputeBuildId_IncludesBuildGeneration(t *testing.T) {
	ld1 := &llamadeployv1.LlamaDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "my-app"},
		Spec:       llamadeployv1.LlamaDeploymentSpec{GitSha: "abc123", BuildGeneration: 0},
	}
	ld2 := &llamadeployv1.LlamaDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "my-app"},
		Spec:       llamadeployv1.LlamaDeploymentSpec{GitSha: "abc123", BuildGeneration: 1},
	}
	id1 := computeBuildId(ld1)
	id2 := computeBuildId(ld2)
	if id1 == id2 {
		t.Error("expected different build IDs when buildGeneration differs")
	}
}

func TestComputeBuildId_DoesNotIncludeImageTag(t *testing.T) {
	ld1 := &llamadeployv1.LlamaDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "my-app"},
		Spec:       llamadeployv1.LlamaDeploymentSpec{GitSha: "abc123", ImageTag: "v1"},
	}
	ld2 := &llamadeployv1.LlamaDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "my-app"},
		Spec:       llamadeployv1.LlamaDeploymentSpec{GitSha: "abc123", ImageTag: "v2"},
	}
	if computeBuildId(ld1) != computeBuildId(ld2) {
		t.Error("expected same build ID regardless of imageTag")
	}
}

// ---------------------------------------------------------------------------
// createBuildJob shape tests
// ---------------------------------------------------------------------------

func TestCreateBuildJob_JobNameTruncation(t *testing.T) {
	r := &LlamaDeploymentReconciler{}
	longName := strings.Repeat("a", 60)
	ld := &llamadeployv1.LlamaDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: longName, Namespace: "default"},
		Spec:       llamadeployv1.LlamaDeploymentSpec{GitSha: "abc123"},
	}
	job := r.createBuildJob(ld, "abcdef1234567890")
	if len(job.Name) > 63 {
		t.Errorf("job name exceeds 63 chars: %q (%d chars)", job.Name, len(job.Name))
	}
}

func TestCreateBuildJob_Labels(t *testing.T) {
	r := &LlamaDeploymentReconciler{}
	ld := &llamadeployv1.LlamaDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "my-app", Namespace: "default"},
		Spec:       llamadeployv1.LlamaDeploymentSpec{GitSha: "abc123"},
	}
	job := r.createBuildJob(ld, "build123")

	if job.Labels["deploy.llamaindex.ai/deployment"] != "my-app" {
		t.Errorf("expected deployment label my-app, got %q", job.Labels["deploy.llamaindex.ai/deployment"])
	}
	if job.Labels["deploy.llamaindex.ai/build-id"] != "build123" {
		t.Errorf("expected build-id label build123, got %q", job.Labels["deploy.llamaindex.ai/build-id"])
	}
}

const envBuildID = "LLAMA_DEPLOY_BUILD_ID"

func TestCreateBuildJob_HasBuildIdEnvVar(t *testing.T) {
	r := &LlamaDeploymentReconciler{}
	ld := &llamadeployv1.LlamaDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "my-app", Namespace: "default"},
		Spec:       llamadeployv1.LlamaDeploymentSpec{GitSha: "abc123"},
	}
	job := r.createBuildJob(ld, "build123")

	found := false
	for _, env := range job.Spec.Template.Spec.Containers[0].Env {
		if env.Name == envBuildID {
			found = true
			if env.Value != "build123" {
				t.Errorf("expected LLAMA_DEPLOY_BUILD_ID=build123, got %q", env.Value)
			}
		}
	}
	if !found {
		t.Error("expected LLAMA_DEPLOY_BUILD_ID env var in build job")
	}
}

// ---------------------------------------------------------------------------
// findContainerResources tests
// ---------------------------------------------------------------------------

func TestFindContainerResources(t *testing.T) {
	containers := []corev1.Container{
		{
			Name: "app",
			Resources: corev1.ResourceRequirements{
				Requests: corev1.ResourceList{corev1.ResourceCPU: resource.MustParse("1")},
			},
		},
		{Name: "sidecar"},
	}

	t.Run("found with resources", func(t *testing.T) {
		res := findContainerResources(containers, "app")
		if res == nil {
			t.Fatal("expected non-nil resources for app container")
		}
	})

	t.Run("found without resources", func(t *testing.T) {
		res := findContainerResources(containers, "sidecar")
		if res != nil {
			t.Error("expected nil for container with no resources")
		}
	})

	t.Run("not found", func(t *testing.T) {
		res := findContainerResources(containers, "nonexistent")
		if res != nil {
			t.Error("expected nil for nonexistent container")
		}
	})
}

// ---------------------------------------------------------------------------
// mergeResourceRequirements tests
// ---------------------------------------------------------------------------

func TestMergeResourceRequirements(t *testing.T) {
	overlay := &corev1.ResourceRequirements{
		Requests: corev1.ResourceList{
			corev1.ResourceCPU: resource.MustParse("2"),
		},
		Limits: corev1.ResourceList{
			corev1.ResourceMemory: resource.MustParse("8Gi"),
		},
	}
	defaults := &corev1.ResourceRequirements{
		Requests: corev1.ResourceList{
			corev1.ResourceCPU:    resource.MustParse("750m"),
			corev1.ResourceMemory: resource.MustParse("2Gi"),
		},
		Limits: corev1.ResourceList{
			corev1.ResourceMemory: resource.MustParse("4Gi"),
		},
	}

	merged := mergeResourceRequirements(overlay, defaults)

	// Overlay CPU should win
	cpuReq := merged.Requests[corev1.ResourceCPU]
	if cpuReq.Cmp(resource.MustParse("2")) != 0 {
		t.Errorf("expected CPU request 2, got %s", cpuReq.String())
	}

	// Default memory request should be backfilled
	memReq := merged.Requests[corev1.ResourceMemory]
	if memReq.Cmp(resource.MustParse("2Gi")) != 0 {
		t.Errorf("expected memory request 2Gi, got %s", memReq.String())
	}

	// Overlay memory limit should win
	memLimit := merged.Limits[corev1.ResourceMemory]
	if memLimit.Cmp(resource.MustParse("8Gi")) != 0 {
		t.Errorf("expected memory limit 8Gi, got %s", memLimit.String())
	}
}

func TestReconcileBuild_StaleJobDeletion_OnGenerationAdvance(t *testing.T) {
	scheme := newTestScheme()

	llamaDeploy := &llamadeployv1.LlamaDeployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:       "my-app",
			Namespace:  "default",
			Generation: 2,
		},
		Spec: llamadeployv1.LlamaDeploymentSpec{
			ProjectId: "proj-123",
			RepoUrl:   "https://github.com/example/repo",
			GitRef:    "main",
			GitSha:    "abc123",
		},
		Status: llamadeployv1.LlamaDeploymentStatus{
			FailedRolloutGeneration: 1,
		},
	}

	buildId := computeBuildId(llamaDeploy)
	jobName := fmt.Sprintf("%s-build-%s", llamaDeploy.Name, buildId)
	if len(jobName) > 63 {
		jobName = jobName[:63]
	}

	existingJob := &batchv1.Job{
		ObjectMeta: metav1.ObjectMeta{
			Name:      jobName,
			Namespace: "default",
		},
		Status: batchv1.JobStatus{
			Failed: 1,
		},
	}

	fakeClient := fake.NewClientBuilder().
		WithScheme(scheme).
		WithObjects(llamaDeploy, existingJob).
		WithStatusSubresource(llamaDeploy).
		Build()

	r := &LlamaDeploymentReconciler{
		Client: fakeClient,
		Scheme: scheme,
	}

	ctx := context.Background()
	_, result, err := r.reconcileBuild(ctx, llamaDeploy)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result == nil {
		t.Fatal("expected non-nil result for requeue")
	}
	if result.RequeueAfter == 0 {
		t.Errorf("expected RequeueAfter > 0, got %v", result.RequeueAfter)
	}

	// Verify the Job was deleted
	var fetchedJob batchv1.Job
	err = fakeClient.Get(ctx, types.NamespacedName{Name: jobName, Namespace: "default"}, &fetchedJob)
	if err == nil {
		t.Errorf("expected Job to be deleted, but it still exists")
	}

	// Verify failedRolloutGeneration was updated to current generation
	// so that if the retry also fails, it stops instead of looping
	var updated llamadeployv1.LlamaDeployment
	if err := fakeClient.Get(ctx, types.NamespacedName{Name: "my-app", Namespace: "default"}, &updated); err != nil {
		t.Fatalf("failed to re-read LlamaDeployment: %v", err)
	}
	if updated.Status.FailedRolloutGeneration != llamaDeploy.Generation {
		t.Errorf("expected failedRolloutGeneration=%d after retry, got %d",
			llamaDeploy.Generation, updated.Status.FailedRolloutGeneration)
	}
}

func TestReconcileBuild_NoJobDeletion_OnSameGeneration(t *testing.T) {
	scheme := newTestScheme()

	llamaDeploy := &llamadeployv1.LlamaDeployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:       "my-app",
			Namespace:  "default",
			Generation: 1,
		},
		Spec: llamadeployv1.LlamaDeploymentSpec{
			ProjectId: "proj-123",
			RepoUrl:   "https://github.com/example/repo",
			GitRef:    "main",
			GitSha:    "abc123",
		},
		Status: llamadeployv1.LlamaDeploymentStatus{
			FailedRolloutGeneration: 1,
		},
	}

	buildId := computeBuildId(llamaDeploy)
	jobName := fmt.Sprintf("%s-build-%s", llamaDeploy.Name, buildId)
	if len(jobName) > 63 {
		jobName = jobName[:63]
	}

	existingJob := &batchv1.Job{
		ObjectMeta: metav1.ObjectMeta{
			Name:      jobName,
			Namespace: "default",
		},
		Status: batchv1.JobStatus{
			Failed: 1,
		},
	}

	fakeClient := fake.NewClientBuilder().
		WithScheme(scheme).
		WithObjects(llamaDeploy, existingJob).
		WithStatusSubresource(llamaDeploy).
		Build()

	r := &LlamaDeploymentReconciler{
		Client: fakeClient,
		Scheme: scheme,
	}

	ctx := context.Background()
	_, result, err := r.reconcileBuild(ctx, llamaDeploy)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result == nil {
		t.Fatal("expected non-nil result")
	}
	expected := ctrl.Result{}
	if *result != expected {
		t.Errorf("expected empty Result (no requeue), got %+v", *result)
	}

	// Verify the phase is BuildFailed
	if llamaDeploy.Status.Phase != PhaseBuildFailed {
		t.Errorf("expected phase %q, got %q", PhaseBuildFailed, llamaDeploy.Status.Phase)
	}

	// Verify the Job still exists
	var fetchedJob batchv1.Job
	err = fakeClient.Get(ctx, types.NamespacedName{Name: jobName, Namespace: "default"}, &fetchedJob)
	if err != nil {
		t.Errorf("expected Job to still exist, but got error: %v", err)
	}
}

func TestInitializeStatus_PreservesPhaseBuilding(t *testing.T) {
	scheme := newTestScheme()

	llamaDeploy := &llamadeployv1.LlamaDeployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:       "my-app",
			Namespace:  "default",
			Generation: 2,
		},
		Spec: llamadeployv1.LlamaDeploymentSpec{
			ProjectId: "proj-123",
			RepoUrl:   "https://github.com/example/repo",
			GitSha:    "abc123",
		},
		Status: llamadeployv1.LlamaDeploymentStatus{
			Phase:                    PhaseBuilding,
			LastReconciledGeneration: 1, // generation changed → needsFullReconcile
			SchemaVersion:            CurrentSchemaVersion,
			AuthToken:                "existing-token",
		},
	}

	fakeClient := fake.NewClientBuilder().
		WithScheme(scheme).
		WithObjects(llamaDeploy).
		WithStatusSubresource(llamaDeploy).
		Build()

	r := &LlamaDeploymentReconciler{
		Client: fakeClient,
		Scheme: scheme,
	}

	ctx := context.Background()
	err := r.initializeStatus(ctx, llamaDeploy, true /* needsFullReconcile */)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Phase should still be Building, not reset to Pending
	if llamaDeploy.Status.Phase != PhaseBuilding {
		t.Errorf("expected phase %q to be preserved, got %q", PhaseBuilding, llamaDeploy.Status.Phase)
	}
}

func TestIsRollingPhase(t *testing.T) {
	// isRollingPhase mirrors the predicate logic in the rollout-aware self-watch
	isRollingPhase := func(phase string) bool {
		return phase == PhasePending || phase == PhaseRollingOut || phase == PhaseBuilding
	}

	tests := []struct {
		phase    string
		expected bool
	}{
		{PhasePending, true},
		{PhaseRollingOut, true},
		{PhaseBuilding, true},
		{PhaseRunning, false},
		{PhaseFailed, false},
		{PhaseRolloutFailed, false},
		{PhaseBuildFailed, false},
		{PhaseSuspended, false},
		{"", false},
	}

	for _, tt := range tests {
		got := isRollingPhase(tt.phase)
		if got != tt.expected {
			t.Errorf("isRollingPhase(%q) = %v, want %v", tt.phase, got, tt.expected)
		}
	}

	// Verify the predicate would fire on transitions OUT of rolling phases
	transitions := []struct {
		oldPhase, newPhase string
		shouldFire         bool
	}{
		{PhaseBuilding, PhaseRunning, true},         // build complete
		{PhaseBuilding, PhaseBuildFailed, true},     // build failed
		{PhasePending, PhaseRunning, true},          // rollout complete
		{PhaseRollingOut, PhaseRunning, true},       // rollout complete
		{PhaseRollingOut, PhaseRolloutFailed, true}, // rollout failed
		{PhaseRunning, PhaseSuspended, false},       // not a rolling phase transition
		{PhaseBuilding, PhasePending, false},        // still in rolling phase
		{PhasePending, PhaseBuilding, false},        // still in rolling phase
	}

	for _, tt := range transitions {
		fires := isRollingPhase(tt.oldPhase) && !isRollingPhase(tt.newPhase)
		if fires != tt.shouldFire {
			t.Errorf("transition %q→%q: fires=%v, want %v", tt.oldPhase, tt.newPhase, fires, tt.shouldFire)
		}
	}
}

func TestReconcileBuild_SkipsSuspendedDeployment(t *testing.T) {
	scheme := newTestScheme()

	ld := &llamadeployv1.LlamaDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "my-app", Namespace: "default"},
		Spec: llamadeployv1.LlamaDeploymentSpec{
			ProjectId:       "proj-123",
			RepoUrl:         "https://github.com/example/repo",
			GitSha:          "abc123",
			Suspended:       true,
			BuildGeneration: 1,
		},
		Status: llamadeployv1.LlamaDeploymentStatus{
			LastBuiltGeneration: 1, // equal to spec.buildGeneration
		},
	}

	fakeClient := fake.NewClientBuilder().
		WithScheme(scheme).
		WithObjects(ld).
		WithStatusSubresource(ld).
		Build()

	r := &LlamaDeploymentReconciler{Client: fakeClient, Scheme: scheme}
	ctx := context.Background()

	buildId, result, err := r.reconcileBuild(ctx, ld)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if buildId != "" {
		t.Errorf("expected empty buildId for suspended deployment, got %q", buildId)
	}
	if result != nil {
		t.Errorf("expected nil result for suspended deployment, got %+v", result)
	}
}

func TestReconcileBuild_CacheHitSucceeded(t *testing.T) {
	scheme := newTestScheme()

	ld := &llamadeployv1.LlamaDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "my-app", Namespace: "default"},
		Spec: llamadeployv1.LlamaDeploymentSpec{
			ProjectId: "proj-123",
			RepoUrl:   "https://github.com/example/repo",
			GitSha:    "abc123",
		},
	}

	expectedBuildId := computeBuildId(ld)
	ld.Status = llamadeployv1.LlamaDeploymentStatus{
		BuildId:     expectedBuildId,
		BuildStatus: BuildStatusSucceeded,
	}

	fakeClient := fake.NewClientBuilder().
		WithScheme(scheme).
		WithObjects(ld).
		WithStatusSubresource(ld).
		Build()

	r := &LlamaDeploymentReconciler{Client: fakeClient, Scheme: scheme}
	ctx := context.Background()

	buildId, result, err := r.reconcileBuild(ctx, ld)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if buildId != expectedBuildId {
		t.Errorf("expected buildId %q, got %q", expectedBuildId, buildId)
	}
	if result != nil {
		t.Errorf("expected nil result for cache hit, got %+v", result)
	}
}

func TestReconcileBuild_JobRunning(t *testing.T) {
	scheme := newTestScheme()

	ld := &llamadeployv1.LlamaDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "my-app", Namespace: "default"},
		Spec: llamadeployv1.LlamaDeploymentSpec{
			ProjectId: "proj-123",
			RepoUrl:   "https://github.com/example/repo",
			GitSha:    "abc123",
		},
		Status: llamadeployv1.LlamaDeploymentStatus{
			AuthToken: "tok",
		},
	}

	buildId := computeBuildId(ld)
	jobName := fmt.Sprintf("%s-build-%s", ld.Name, buildId)
	if len(jobName) > 63 {
		jobName = jobName[:63]
	}

	// Seed a running Job (no Succeeded/Failed)
	existingJob := &batchv1.Job{
		ObjectMeta: metav1.ObjectMeta{
			Name:      jobName,
			Namespace: "default",
		},
		Status: batchv1.JobStatus{},
	}

	fakeClient := fake.NewClientBuilder().
		WithScheme(scheme).
		WithObjects(ld, existingJob).
		WithStatusSubresource(ld).
		Build()

	r := &LlamaDeploymentReconciler{Client: fakeClient, Scheme: scheme}
	ctx := context.Background()

	gotBuildId, result, err := r.reconcileBuild(ctx, ld)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if gotBuildId != "" {
		t.Errorf("expected empty buildId for running job, got %q", gotBuildId)
	}
	if result == nil {
		t.Fatal("expected non-nil result for running job")
	}
	if result.RequeueAfter == 0 {
		t.Error("expected RequeueAfter > 0 for running job")
	}
}

func TestInitializeStatus_ResetsPendingPhase(t *testing.T) {
	scheme := newTestScheme()

	llamaDeploy := &llamadeployv1.LlamaDeployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:       "my-app",
			Namespace:  "default",
			Generation: 2,
		},
		Spec: llamadeployv1.LlamaDeploymentSpec{
			ProjectId: "proj-123",
			RepoUrl:   "https://github.com/example/repo",
		},
		Status: llamadeployv1.LlamaDeploymentStatus{
			Phase:                    PhaseRunning,
			LastReconciledGeneration: 1,
			SchemaVersion:            CurrentSchemaVersion,
			AuthToken:                "existing-token",
		},
	}

	fakeClient := fake.NewClientBuilder().
		WithScheme(scheme).
		WithObjects(llamaDeploy).
		WithStatusSubresource(llamaDeploy).
		Build()

	r := &LlamaDeploymentReconciler{
		Client: fakeClient,
		Scheme: scheme,
	}

	ctx := context.Background()
	err := r.initializeStatus(ctx, llamaDeploy, true /* needsFullReconcile */)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Non-terminal, non-Building phases should be reset to Pending
	if llamaDeploy.Status.Phase != PhasePending {
		t.Errorf("expected phase %q, got %q", PhasePending, llamaDeploy.Status.Phase)
	}
}

func TestReconcileBuild_SkipsWhenRepoUrlEmpty(t *testing.T) {
	scheme := newTestScheme()

	llamaDeploy := &llamadeployv1.LlamaDeployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "pending-app",
			Namespace: "default",
		},
		Spec: llamadeployv1.LlamaDeploymentSpec{
			ProjectId: "proj-123",
			RepoUrl:   "", // no code source configured
		},
	}

	fakeClient := fake.NewClientBuilder().
		WithScheme(scheme).
		WithObjects(llamaDeploy).
		WithStatusSubresource(llamaDeploy).
		Build()

	r := &LlamaDeploymentReconciler{
		Client: fakeClient,
		Scheme: scheme,
	}

	ctx := context.Background()
	buildId, result, err := r.reconcileBuild(ctx, llamaDeploy)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result != nil {
		t.Errorf("expected nil result for skipped build, got %+v", result)
	}
	if buildId != "" {
		t.Errorf("expected empty buildId, got %q", buildId)
	}
}

// When the spec advances mid-build, the in-flight Job for the old buildId
// must be deleted so it doesn't race the new one to upload.
func TestReconcileBuild_SupersedesInFlightJob_OnBuildIdChange(t *testing.T) {
	r, ld, c := newBuildSupersedeFixture(t, buildSupersedeFixture{
		deploymentName: "my-app",
		staleBuildId:   "stalebuild1234",
		newGitSha:      "abc123",
		staleJobStatus: &batchv1.JobStatus{}, // in-flight: no Succeeded, no Failed
	})
	ctx := context.Background()

	_, result, err := r.reconcileBuild(ctx, ld)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result == nil {
		t.Fatal("expected non-nil result after creating new build job")
	}

	// The stale Job should have been deleted.
	staleJobName := buildJobName(ld.Name, "stalebuild1234")
	var fetchedStale batchv1.Job
	if err := c.Get(ctx, types.NamespacedName{Name: staleJobName, Namespace: "default"}, &fetchedStale); err == nil {
		t.Error("expected stale Job to be deleted, but it still exists")
	}

	// The new build Job should exist.
	newBuildId := computeBuildId(ld)
	if newBuildId == "stalebuild1234" {
		t.Fatalf("test invariant broken: stale and new buildId are both %q", newBuildId)
	}
	newJobName := buildJobName(ld.Name, newBuildId)
	var newJob batchv1.Job
	if err := c.Get(ctx, types.NamespacedName{Name: newJobName, Namespace: "default"}, &newJob); err != nil {
		t.Errorf("expected new build Job %q to be created: %v", newJobName, err)
	}
}

// Succeeded stale Jobs must be left alone so their artifacts remain available
// for A → B → A rollback-by-cache-hit.
func TestReconcileBuild_DoesNotDeleteSucceededSupersededJob(t *testing.T) {
	r, ld, c := newBuildSupersedeFixture(t, buildSupersedeFixture{
		deploymentName: "my-app",
		staleBuildId:   "succeededstale1",
		newGitSha:      "new-sha",
		// BuildStatus is Running (stale) but the Job itself has Succeeded —
		// the operator should inspect the Job and leave it alone.
		staleJobStatus: &batchv1.JobStatus{Succeeded: 1},
	})
	ctx := context.Background()

	if _, _, err := r.reconcileBuild(ctx, ld); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	staleJobName := buildJobName(ld.Name, "succeededstale1")
	var fetchedStale batchv1.Job
	if err := c.Get(ctx, types.NamespacedName{Name: staleJobName, Namespace: "default"}, &fetchedStale); err != nil {
		t.Errorf("expected succeeded stale Job to be preserved, got err: %v", err)
	}
}

// A stale Job that's already been reaped by TTL should not error — we just
// proceed to create the new Job.
func TestReconcileBuild_SupersedesJob_WhenStaleJobAlreadyGone(t *testing.T) {
	r, ld, c := newBuildSupersedeFixture(t, buildSupersedeFixture{
		deploymentName: "my-app",
		staleBuildId:   "reapedstale12",
		newGitSha:      "abc123",
		staleJobStatus: nil, // no stale Job in the cluster
	})
	ctx := context.Background()

	_, result, err := r.reconcileBuild(ctx, ld)
	if err != nil {
		t.Fatalf("unexpected error when stale Job is absent: %v", err)
	}
	if result == nil {
		t.Fatal("expected non-nil result when creating new build job")
	}

	newJobName := buildJobName(ld.Name, computeBuildId(ld))
	var newJob batchv1.Job
	if err := c.Get(ctx, types.NamespacedName{Name: newJobName, Namespace: "default"}, &newJob); err != nil {
		t.Errorf("expected new build Job %q to be created: %v", newJobName, err)
	}
}

func TestReconcileBuild_ProceedsWhenRepoUrlSetButGitShaEmpty(t *testing.T) {
	scheme := newTestScheme()

	llamaDeploy := &llamadeployv1.LlamaDeployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "ready-app",
			Namespace: "default",
		},
		Spec: llamadeployv1.LlamaDeploymentSpec{
			ProjectId: "proj-123",
			RepoUrl:   "https://github.com/example/repo",
			GitRef:    "main",
			// GitSha intentionally empty — build should still proceed
		},
	}

	fakeClient := fake.NewClientBuilder().
		WithScheme(scheme).
		WithObjects(llamaDeploy).
		WithStatusSubresource(llamaDeploy).
		Build()

	r := &LlamaDeploymentReconciler{
		Client: fakeClient,
		Scheme: scheme,
	}

	ctx := context.Background()
	_, result, err := r.reconcileBuild(ctx, llamaDeploy)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	// When RepoUrl is set, the build should proceed (create a job and requeue),
	// not skip with nil result like a pending deployment.
	if result == nil {
		t.Fatal("expected non-nil result (build should proceed and requeue), got nil")
	}
	if result.RequeueAfter == 0 {
		t.Error("expected RequeueAfter > 0 indicating build job was created")
	}
	// Verify status was updated to Building
	if llamaDeploy.Status.Phase != PhaseBuilding {
		t.Errorf("expected phase %q, got %q", PhaseBuilding, llamaDeploy.Status.Phase)
	}
}
