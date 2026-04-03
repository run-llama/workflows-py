//go:build !integration

package controller

import (
	"context"
	"testing"
	"time"

	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"

	llamadeployv1 "llama-agents-operator/api/v1"
)

// testSchemeWithApps returns a scheme with appsv1 registered in addition to
// the types from newTestScheme.
func testSchemeWithApps() *runtime.Scheme {
	scheme := newTestScheme()
	_ = appsv1.AddToScheme(scheme)
	return scheme
}

// ---------------------------------------------------------------------------
// assessDeploymentHealth tests
// ---------------------------------------------------------------------------

func TestAssessDeploymentHealth_RolloutTimeout(t *testing.T) {
	scheme := testSchemeWithApps()

	// Seed a Deployment with 0 available replicas so timeout yields PhaseFailed.
	dep := &appsv1.Deployment{
		ObjectMeta: metav1.ObjectMeta{Name: "my-app", Namespace: "default"},
		Status: appsv1.DeploymentStatus{
			AvailableReplicas: 0,
			Conditions: []appsv1.DeploymentCondition{
				{Type: appsv1.DeploymentProgressing, Status: corev1.ConditionTrue},
			},
		},
	}

	expiredTime := metav1.NewTime(time.Now().Add(-2 * time.Hour))
	ld := &llamadeployv1.LlamaDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "my-app", Namespace: "default"},
		Spec:       llamadeployv1.LlamaDeploymentSpec{ProjectId: "p", RepoUrl: "r"},
		Status: llamadeployv1.LlamaDeploymentStatus{
			Phase:            PhasePending,
			RolloutStartedAt: &expiredTime,
			AuthToken:        "tok",
		},
	}

	fakeClient := fake.NewClientBuilder().
		WithScheme(scheme).
		WithObjects(ld, dep).
		WithStatusSubresource(ld).
		Build()

	r := &LlamaDeploymentReconciler{Client: fakeClient, Scheme: scheme}
	ctx := context.Background()

	phase, _, _, _, err := r.assessDeploymentHealth(ctx, ld)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if phase != PhaseFailed {
		t.Errorf("expected phase %q on timeout with 0 available replicas, got %q", PhaseFailed, phase)
	}
}

func TestAssessDeploymentHealth_RolloutTimeout_WithAvailableReplicas(t *testing.T) {
	scheme := testSchemeWithApps()

	// Seed a Deployment with available replicas so timeout yields PhaseRolloutFailed.
	dep := &appsv1.Deployment{
		ObjectMeta: metav1.ObjectMeta{Name: "my-app", Namespace: "default"},
		Status: appsv1.DeploymentStatus{
			AvailableReplicas: 1,
			Conditions: []appsv1.DeploymentCondition{
				{Type: appsv1.DeploymentProgressing, Status: corev1.ConditionTrue},
			},
		},
	}

	expiredTime := metav1.NewTime(time.Now().Add(-2 * time.Hour))
	ld := &llamadeployv1.LlamaDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "my-app", Namespace: "default"},
		Spec:       llamadeployv1.LlamaDeploymentSpec{ProjectId: "p", RepoUrl: "r"},
		Status: llamadeployv1.LlamaDeploymentStatus{
			Phase:            PhaseRollingOut,
			RolloutStartedAt: &expiredTime,
			AuthToken:        "tok",
		},
	}

	fakeClient := fake.NewClientBuilder().
		WithScheme(scheme).
		WithObjects(ld, dep).
		WithStatusSubresource(ld).
		Build()

	r := &LlamaDeploymentReconciler{Client: fakeClient, Scheme: scheme}
	ctx := context.Background()

	phase, _, _, _, err := r.assessDeploymentHealth(ctx, ld)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if phase != PhaseRolloutFailed {
		t.Errorf("expected phase %q on timeout with available replicas, got %q", PhaseRolloutFailed, phase)
	}
}

func TestAssessDeploymentHealth_PhaseRunning(t *testing.T) {
	scheme := testSchemeWithApps()

	dep := &appsv1.Deployment{
		ObjectMeta: metav1.ObjectMeta{Name: "my-app", Namespace: "default"},
		Spec:       appsv1.DeploymentSpec{Replicas: ptr(int32(1))},
		Status: appsv1.DeploymentStatus{
			AvailableReplicas: 1,
			ReadyReplicas:     1,
			Replicas:          1,
			Conditions: []appsv1.DeploymentCondition{
				{Type: appsv1.DeploymentAvailable, Status: corev1.ConditionTrue},
				{Type: appsv1.DeploymentProgressing, Status: corev1.ConditionTrue},
			},
		},
	}

	ld := &llamadeployv1.LlamaDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "my-app", Namespace: "default"},
		Spec:       llamadeployv1.LlamaDeploymentSpec{ProjectId: "p", RepoUrl: "r"},
		Status: llamadeployv1.LlamaDeploymentStatus{
			Phase:     PhaseRollingOut,
			AuthToken: "tok",
		},
	}

	fakeClient := fake.NewClientBuilder().
		WithScheme(scheme).
		WithObjects(ld, dep).
		WithStatusSubresource(ld).
		Build()

	r := &LlamaDeploymentReconciler{Client: fakeClient, Scheme: scheme}
	ctx := context.Background()

	phase, _, _, _, err := r.assessDeploymentHealth(ctx, ld)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if phase != PhaseRunning {
		t.Errorf("expected phase %q, got %q", PhaseRunning, phase)
	}
}

func TestAssessDeploymentHealth_PhasePending_NoDeployment(t *testing.T) {
	scheme := testSchemeWithApps()

	ld := &llamadeployv1.LlamaDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "my-app", Namespace: "default"},
		Spec:       llamadeployv1.LlamaDeploymentSpec{ProjectId: "p", RepoUrl: "r"},
		Status: llamadeployv1.LlamaDeploymentStatus{
			Phase:     PhasePending,
			AuthToken: "tok",
		},
	}

	fakeClient := fake.NewClientBuilder().
		WithScheme(scheme).
		WithObjects(ld).
		WithStatusSubresource(ld).
		Build()

	r := &LlamaDeploymentReconciler{Client: fakeClient, Scheme: scheme}
	ctx := context.Background()

	phase, _, _, statusDirty, err := r.assessDeploymentHealth(ctx, ld)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if phase != PhasePending {
		t.Errorf("expected phase %q, got %q", PhasePending, phase)
	}
	// RolloutStartedAt should be set since phase is Pending and it was nil
	if !statusDirty {
		t.Error("expected statusDirty=true when RolloutStartedAt is first set")
	}
	if ld.Status.RolloutStartedAt == nil {
		t.Error("expected RolloutStartedAt to be set")
	}
}

func TestAssessDeploymentHealth_ClearsRolloutStartedAt_OnRunning(t *testing.T) {
	scheme := testSchemeWithApps()

	dep := &appsv1.Deployment{
		ObjectMeta: metav1.ObjectMeta{Name: "my-app", Namespace: "default"},
		Spec:       appsv1.DeploymentSpec{Replicas: ptr(int32(1))},
		Status: appsv1.DeploymentStatus{
			AvailableReplicas: 1,
			ReadyReplicas:     1,
			Replicas:          1,
			Conditions: []appsv1.DeploymentCondition{
				{Type: appsv1.DeploymentAvailable, Status: corev1.ConditionTrue},
				{Type: appsv1.DeploymentProgressing, Status: corev1.ConditionTrue},
			},
		},
	}

	startTime := metav1.NewTime(time.Now().Add(-1 * time.Minute))
	ld := &llamadeployv1.LlamaDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "my-app", Namespace: "default"},
		Spec:       llamadeployv1.LlamaDeploymentSpec{ProjectId: "p", RepoUrl: "r"},
		Status: llamadeployv1.LlamaDeploymentStatus{
			Phase:            PhaseRollingOut,
			RolloutStartedAt: &startTime,
			AuthToken:        "tok",
		},
	}

	fakeClient := fake.NewClientBuilder().
		WithScheme(scheme).
		WithObjects(ld, dep).
		WithStatusSubresource(ld).
		Build()

	r := &LlamaDeploymentReconciler{Client: fakeClient, Scheme: scheme}
	ctx := context.Background()

	phase, _, _, statusDirty, err := r.assessDeploymentHealth(ctx, ld)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if phase != PhaseRunning {
		t.Errorf("expected phase %q, got %q", PhaseRunning, phase)
	}
	if !statusDirty {
		t.Error("expected statusDirty=true when RolloutStartedAt is cleared")
	}
	if ld.Status.RolloutStartedAt != nil {
		t.Error("expected RolloutStartedAt to be cleared on Running phase")
	}
}

func TestAssessDeploymentHealth_PhaseSuspended(t *testing.T) {
	scheme := testSchemeWithApps()

	ld := &llamadeployv1.LlamaDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "my-app", Namespace: "default"},
		Spec:       llamadeployv1.LlamaDeploymentSpec{ProjectId: "p", RepoUrl: "r", Suspended: true},
		Status: llamadeployv1.LlamaDeploymentStatus{
			Phase:     PhaseRunning,
			AuthToken: "tok",
		},
	}

	fakeClient := fake.NewClientBuilder().
		WithScheme(scheme).
		WithObjects(ld).
		WithStatusSubresource(ld).
		Build()

	r := &LlamaDeploymentReconciler{Client: fakeClient, Scheme: scheme}
	ctx := context.Background()

	phase, _, _, _, err := r.assessDeploymentHealth(ctx, ld)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if phase != PhaseSuspended {
		t.Errorf("expected phase %q for suspended deployment, got %q", PhaseSuspended, phase)
	}
}

func TestAssessDeploymentHealth_PhaseFailed_ProgressFalse_NoAvailable(t *testing.T) {
	scheme := testSchemeWithApps()

	dep := &appsv1.Deployment{
		ObjectMeta: metav1.ObjectMeta{Name: "my-app", Namespace: "default"},
		Status: appsv1.DeploymentStatus{
			AvailableReplicas: 0,
			Conditions: []appsv1.DeploymentCondition{
				{Type: appsv1.DeploymentProgressing, Status: corev1.ConditionFalse},
			},
		},
	}

	ld := &llamadeployv1.LlamaDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "my-app", Namespace: "default"},
		Spec:       llamadeployv1.LlamaDeploymentSpec{ProjectId: "p", RepoUrl: "r"},
		Status: llamadeployv1.LlamaDeploymentStatus{
			Phase:     PhasePending,
			AuthToken: "tok",
		},
	}

	fakeClient := fake.NewClientBuilder().
		WithScheme(scheme).
		WithObjects(ld, dep).
		WithStatusSubresource(ld).
		Build()

	r := &LlamaDeploymentReconciler{Client: fakeClient, Scheme: scheme}
	ctx := context.Background()

	phase, _, _, _, err := r.assessDeploymentHealth(ctx, ld)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if phase != PhaseFailed {
		t.Errorf("expected phase %q, got %q", PhaseFailed, phase)
	}
}

// ---------------------------------------------------------------------------
// checkCapacityGates tests
// ---------------------------------------------------------------------------

func TestCheckCapacityGates_SkipsWhenNotNeedsFullReconcile(t *testing.T) {
	scheme := testSchemeWithApps()

	ld := &llamadeployv1.LlamaDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "my-app", Namespace: "default"},
		Spec:       llamadeployv1.LlamaDeploymentSpec{ProjectId: "p", RepoUrl: "r"},
	}

	fakeClient := fake.NewClientBuilder().
		WithScheme(scheme).
		WithObjects(ld).
		Build()

	r := &LlamaDeploymentReconciler{
		Client:         fakeClient,
		Scheme:         scheme,
		MaxDeployments: 1,
	}
	ctx := context.Background()

	result, err := r.checkCapacityGates(ctx, ld, false)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result != nil {
		t.Errorf("expected nil result when needsFullReconcile=false, got %+v", result)
	}
}

func TestCheckCapacityGates_SkipsWhenSuspended(t *testing.T) {
	scheme := testSchemeWithApps()

	ld := &llamadeployv1.LlamaDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "my-app", Namespace: "default"},
		Spec:       llamadeployv1.LlamaDeploymentSpec{ProjectId: "p", RepoUrl: "r", Suspended: true},
	}

	fakeClient := fake.NewClientBuilder().
		WithScheme(scheme).
		WithObjects(ld).
		Build()

	r := &LlamaDeploymentReconciler{
		Client:         fakeClient,
		Scheme:         scheme,
		MaxDeployments: 1,
	}
	ctx := context.Background()

	result, err := r.checkCapacityGates(ctx, ld, true)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result != nil {
		t.Errorf("expected nil result when Suspended=true, got %+v", result)
	}
}

func TestCheckCapacityGates_SkipsWhenAwaitingCodePhase(t *testing.T) {
	scheme := testSchemeWithApps()

	// Deployment already in PhaseAwaitingCode (has a RepoUrl now but phase hasn't transitioned yet)
	ld := &llamadeployv1.LlamaDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "new-app", Namespace: "default"},
		Spec:       llamadeployv1.LlamaDeploymentSpec{ProjectId: "p", RepoUrl: ""},
		Status: llamadeployv1.LlamaDeploymentStatus{
			Phase: PhaseAwaitingCode,
		},
	}

	// An existing active deployment that fills the capacity
	existing := &llamadeployv1.LlamaDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "existing-app", Namespace: "default"},
		Spec:       llamadeployv1.LlamaDeploymentSpec{ProjectId: "p2", RepoUrl: "r2"},
		Status: llamadeployv1.LlamaDeploymentStatus{
			Phase: PhaseRunning,
		},
	}

	fakeClient := fake.NewClientBuilder().
		WithScheme(scheme).
		WithObjects(ld, existing).
		WithStatusSubresource(ld, existing).
		Build()

	r := &LlamaDeploymentReconciler{
		Client:                fakeClient,
		Scheme:                scheme,
		MaxDeployments:        1,
		MaxConcurrentRollouts: 1,
	}
	ctx := context.Background()

	result, err := r.checkCapacityGates(ctx, ld, true)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result != nil {
		t.Errorf("expected nil result when phase is AwaitingCode, got %+v", result)
	}
}

func TestCheckCapacityGates_RequeuesWhenMaxDeploymentsExceeded(t *testing.T) {
	scheme := testSchemeWithApps()

	// Current deployment being reconciled
	ld := &llamadeployv1.LlamaDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "new-app", Namespace: "default"},
		Spec:       llamadeployv1.LlamaDeploymentSpec{ProjectId: "p", RepoUrl: "r"},
	}

	// An existing active deployment that fills the capacity
	existing := &llamadeployv1.LlamaDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "existing-app", Namespace: "default"},
		Spec:       llamadeployv1.LlamaDeploymentSpec{ProjectId: "p2", RepoUrl: "r2"},
		Status: llamadeployv1.LlamaDeploymentStatus{
			Phase: PhaseRunning,
		},
	}

	fakeClient := fake.NewClientBuilder().
		WithScheme(scheme).
		WithObjects(ld, existing).
		WithStatusSubresource(ld, existing).
		Build()

	r := &LlamaDeploymentReconciler{
		Client:         fakeClient,
		Scheme:         scheme,
		MaxDeployments: 1,
	}
	ctx := context.Background()

	result, err := r.checkCapacityGates(ctx, ld, true)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result == nil {
		t.Fatal("expected non-nil result when max deployments exceeded")
	}
	if result.RequeueAfter == 0 {
		t.Error("expected RequeueAfter > 0")
	}
}

// ---------------------------------------------------------------------------
// checkSecretGate tests
// ---------------------------------------------------------------------------

func TestCheckSecretGate_NilWhenSecretNameEmpty(t *testing.T) {
	scheme := testSchemeWithApps()

	ld := &llamadeployv1.LlamaDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "my-app", Namespace: "default"},
		Spec:       llamadeployv1.LlamaDeploymentSpec{ProjectId: "p", RepoUrl: "r", SecretName: ""},
	}

	fakeClient := fake.NewClientBuilder().
		WithScheme(scheme).
		WithObjects(ld).
		Build()

	r := &LlamaDeploymentReconciler{Client: fakeClient, Scheme: scheme}
	ctx := context.Background()

	result, err := r.checkSecretGate(ctx, ld)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result != nil {
		t.Errorf("expected nil result when SecretName is empty, got %+v", result)
	}
}

// ---------------------------------------------------------------------------
// finalizePhase tests
// ---------------------------------------------------------------------------

func TestFinalizePhase_NoWriteWhenUnchangedAndNotDirty(t *testing.T) {
	scheme := testSchemeWithApps()

	ld := &llamadeployv1.LlamaDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "my-app", Namespace: "default"},
		Spec:       llamadeployv1.LlamaDeploymentSpec{ProjectId: "p", RepoUrl: "r"},
		Status: llamadeployv1.LlamaDeploymentStatus{
			Phase:     PhaseRunning,
			Message:   "all good",
			AuthToken: "tok",
		},
	}

	fakeClient := fake.NewClientBuilder().
		WithScheme(scheme).
		WithObjects(ld).
		WithStatusSubresource(ld).
		Build()

	r := &LlamaDeploymentReconciler{Client: fakeClient, Scheme: scheme}
	ctx := context.Background()

	// Same phase, statusDirty=false → should not write
	result, err := r.finalizePhase(ctx, ld, PhaseRunning, "all good", 0, false)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.RequeueAfter != 0 {
		t.Errorf("expected no requeue, got %v", result.RequeueAfter)
	}
	// LastUpdated should not have been set since we didn't enter the update path
	if ld.Status.LastUpdated != nil {
		t.Error("expected LastUpdated to remain nil when no write happens")
	}
}

func TestFinalizePhase_WritesOnPhaseChange(t *testing.T) {
	scheme := testSchemeWithApps()

	ld := &llamadeployv1.LlamaDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "my-app", Namespace: "default"},
		Spec:       llamadeployv1.LlamaDeploymentSpec{ProjectId: "p", RepoUrl: "r"},
		Status: llamadeployv1.LlamaDeploymentStatus{
			Phase:     PhasePending,
			AuthToken: "tok",
		},
	}

	fakeClient := fake.NewClientBuilder().
		WithScheme(scheme).
		WithObjects(ld).
		WithStatusSubresource(ld).
		Build()

	r := &LlamaDeploymentReconciler{Client: fakeClient, Scheme: scheme}
	ctx := context.Background()

	_, err := r.finalizePhase(ctx, ld, PhaseRunning, "deployment healthy", 0, false)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if ld.Status.Phase != PhaseRunning {
		t.Errorf("expected phase %q, got %q", PhaseRunning, ld.Status.Phase)
	}
	if ld.Status.LastUpdated == nil {
		t.Error("expected LastUpdated to be set on phase change")
	}
}

func TestFinalizePhase_WritesWhenStatusDirty(t *testing.T) {
	scheme := testSchemeWithApps()

	ld := &llamadeployv1.LlamaDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "my-app", Namespace: "default"},
		Spec:       llamadeployv1.LlamaDeploymentSpec{ProjectId: "p", RepoUrl: "r"},
		Status: llamadeployv1.LlamaDeploymentStatus{
			Phase:     PhaseRunning,
			AuthToken: "tok",
		},
	}

	fakeClient := fake.NewClientBuilder().
		WithScheme(scheme).
		WithObjects(ld).
		WithStatusSubresource(ld).
		Build()

	r := &LlamaDeploymentReconciler{Client: fakeClient, Scheme: scheme}
	ctx := context.Background()

	// Same phase but statusDirty=true → should still write
	_, err := r.finalizePhase(ctx, ld, PhaseRunning, "deployment healthy", 0, true)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if ld.Status.LastUpdated == nil {
		t.Error("expected LastUpdated to be set when statusDirty=true")
	}
}

// ---------------------------------------------------------------------------
// isFailedPhase tests
// ---------------------------------------------------------------------------

func TestIsFailedPhase(t *testing.T) {
	tests := []struct {
		phase string
		want  bool
	}{
		{PhaseRolloutFailed, true},
		{PhaseFailed, true},
		{PhaseRunning, false},
		{PhasePending, false},
		{PhaseBuilding, false},
		{PhaseBuildFailed, false},
		{PhaseSuspended, false},
		{PhaseAwaitingCode, false},
		{"", false},
	}
	for _, tt := range tests {
		t.Run(tt.phase, func(t *testing.T) {
			if got := isFailedPhase(tt.phase); got != tt.want {
				t.Errorf("isFailedPhase(%q) = %v, want %v", tt.phase, got, tt.want)
			}
		})
	}
}

// ---------------------------------------------------------------------------
// checkRolloutTimeout tests
// ---------------------------------------------------------------------------

func TestCheckRolloutTimeout_NilRolloutStartedAt(t *testing.T) {
	scheme := testSchemeWithApps()
	ld := &llamadeployv1.LlamaDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "my-app", Namespace: "default"},
		Status: llamadeployv1.LlamaDeploymentStatus{
			RolloutStartedAt: nil,
		},
	}
	fakeClient := fake.NewClientBuilder().WithScheme(scheme).WithObjects(ld).Build()
	r := &LlamaDeploymentReconciler{Client: fakeClient, Scheme: scheme}
	ctx := context.Background()

	result := r.checkRolloutTimeout(ctx, ld)
	if result.TimedOut {
		t.Error("expected TimedOut=false when RolloutStartedAt is nil")
	}
	if result.RequeueAfter != 0 {
		t.Errorf("expected RequeueAfter=0, got %v", result.RequeueAfter)
	}
}

func TestCheckRolloutTimeout_RecentStart_RequeuesWithRemaining(t *testing.T) {
	scheme := testSchemeWithApps()
	recentTime := metav1.NewTime(time.Now().Add(-1 * time.Minute))
	ld := &llamadeployv1.LlamaDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "my-app", Namespace: "default"},
		Status: llamadeployv1.LlamaDeploymentStatus{
			RolloutStartedAt: &recentTime,
		},
	}
	fakeClient := fake.NewClientBuilder().WithScheme(scheme).WithObjects(ld).Build()
	r := &LlamaDeploymentReconciler{Client: fakeClient, Scheme: scheme}
	ctx := context.Background()

	result := r.checkRolloutTimeout(ctx, ld)
	if result.TimedOut {
		t.Error("expected TimedOut=false for recent rollout start")
	}
	if result.RequeueAfter <= 0 {
		t.Error("expected RequeueAfter > 0 for in-progress rollout")
	}
}

// ---------------------------------------------------------------------------
// checkRolloutCapacity tests
// ---------------------------------------------------------------------------

func TestCheckRolloutCapacity_UnlimitedWhenZero(t *testing.T) {
	scheme := testSchemeWithApps()
	ld := &llamadeployv1.LlamaDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "my-app", Namespace: "default"},
	}
	fakeClient := fake.NewClientBuilder().WithScheme(scheme).WithObjects(ld).Build()
	r := &LlamaDeploymentReconciler{Client: fakeClient, Scheme: scheme, MaxConcurrentRollouts: 0}
	ctx := context.Background()

	requeue, _, err := r.checkRolloutCapacity(ctx, ld)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if requeue {
		t.Error("expected no requeue when MaxConcurrentRollouts=0")
	}
}

func TestCheckRolloutCapacity_RequeuesAtLimit(t *testing.T) {
	scheme := testSchemeWithApps()
	current := &llamadeployv1.LlamaDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "new-app", Namespace: "default"},
	}
	rolling := &llamadeployv1.LlamaDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "rolling-app", Namespace: "default"},
		Spec:       llamadeployv1.LlamaDeploymentSpec{ProjectId: "p"},
		Status:     llamadeployv1.LlamaDeploymentStatus{Phase: PhaseRollingOut},
	}
	fakeClient := fake.NewClientBuilder().
		WithScheme(scheme).
		WithObjects(current, rolling).
		WithStatusSubresource(current, rolling).
		Build()
	r := &LlamaDeploymentReconciler{Client: fakeClient, Scheme: scheme, MaxConcurrentRollouts: 1}
	ctx := context.Background()

	requeue, result, err := r.checkRolloutCapacity(ctx, current)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !requeue {
		t.Error("expected requeue when at rollout capacity")
	}
	if result.RequeueAfter == 0 {
		t.Error("expected RequeueAfter > 0")
	}
}

func TestCheckRolloutCapacity_IgnoresSuspendedDeployments(t *testing.T) {
	scheme := testSchemeWithApps()
	current := &llamadeployv1.LlamaDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "new-app", Namespace: "default"},
	}
	// Suspended deployment in Pending phase should NOT count toward limit
	suspended := &llamadeployv1.LlamaDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "suspended-app", Namespace: "default"},
		Spec:       llamadeployv1.LlamaDeploymentSpec{ProjectId: "p", Suspended: true},
		Status:     llamadeployv1.LlamaDeploymentStatus{Phase: PhasePending},
	}
	fakeClient := fake.NewClientBuilder().
		WithScheme(scheme).
		WithObjects(current, suspended).
		WithStatusSubresource(current, suspended).
		Build()
	r := &LlamaDeploymentReconciler{Client: fakeClient, Scheme: scheme, MaxConcurrentRollouts: 1}
	ctx := context.Background()

	requeue, _, err := r.checkRolloutCapacity(ctx, current)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if requeue {
		t.Error("expected no requeue when suspended deployments don't count")
	}
}

func TestCheckRolloutCapacity_IgnoresAwaitingCodeDeployments(t *testing.T) {
	scheme := testSchemeWithApps()
	current := &llamadeployv1.LlamaDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "new-app", Namespace: "default"},
	}
	// Deployment in AwaitingCode phase should NOT count toward rollout limit
	waitingForPush := &llamadeployv1.LlamaDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "waiting-app", Namespace: "default"},
		Spec:       llamadeployv1.LlamaDeploymentSpec{ProjectId: "p"},
		Status:     llamadeployv1.LlamaDeploymentStatus{Phase: PhaseAwaitingCode},
	}
	fakeClient := fake.NewClientBuilder().
		WithScheme(scheme).
		WithObjects(current, waitingForPush).
		WithStatusSubresource(current, waitingForPush).
		Build()
	r := &LlamaDeploymentReconciler{Client: fakeClient, Scheme: scheme, MaxConcurrentRollouts: 1}
	ctx := context.Background()

	requeue, _, err := r.checkRolloutCapacity(ctx, current)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if requeue {
		t.Error("expected no requeue when AwaitingCode deployments don't count")
	}
}

// ---------------------------------------------------------------------------
// checkDeploymentCapacity tests
// ---------------------------------------------------------------------------

func TestCheckDeploymentCapacity_UnlimitedWhenZero(t *testing.T) {
	scheme := testSchemeWithApps()
	ld := &llamadeployv1.LlamaDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "my-app", Namespace: "default"},
	}
	fakeClient := fake.NewClientBuilder().WithScheme(scheme).WithObjects(ld).Build()
	r := &LlamaDeploymentReconciler{Client: fakeClient, Scheme: scheme, MaxDeployments: 0}
	ctx := context.Background()

	requeue, _, err := r.checkDeploymentCapacity(ctx, ld)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if requeue {
		t.Error("expected no requeue when MaxDeployments=0")
	}
}

func TestCheckDeploymentCapacity_RequeuesAtLimit(t *testing.T) {
	scheme := testSchemeWithApps()
	current := &llamadeployv1.LlamaDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "new-app", Namespace: "default"},
	}
	existing := &llamadeployv1.LlamaDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "existing-app", Namespace: "default"},
		Status:     llamadeployv1.LlamaDeploymentStatus{Phase: PhaseRunning},
	}
	fakeClient := fake.NewClientBuilder().
		WithScheme(scheme).
		WithObjects(current, existing).
		WithStatusSubresource(current, existing).
		Build()
	r := &LlamaDeploymentReconciler{Client: fakeClient, Scheme: scheme, MaxDeployments: 1}
	ctx := context.Background()

	requeue, result, err := r.checkDeploymentCapacity(ctx, current)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !requeue {
		t.Error("expected requeue when at deployment capacity")
	}
	if result.RequeueAfter == 0 {
		t.Error("expected RequeueAfter > 0")
	}
}

func TestCheckDeploymentCapacity_IgnoresAwaitingCodeDeployments(t *testing.T) {
	scheme := testSchemeWithApps()
	current := &llamadeployv1.LlamaDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "new-app", Namespace: "default"},
	}
	// Deployment in AwaitingCode phase should NOT count toward deployment limit
	waitingForPush := &llamadeployv1.LlamaDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "waiting-app", Namespace: "default"},
		Spec:       llamadeployv1.LlamaDeploymentSpec{ProjectId: "p"},
		Status:     llamadeployv1.LlamaDeploymentStatus{Phase: PhaseAwaitingCode},
	}
	fakeClient := fake.NewClientBuilder().
		WithScheme(scheme).
		WithObjects(current, waitingForPush).
		WithStatusSubresource(current, waitingForPush).
		Build()
	r := &LlamaDeploymentReconciler{Client: fakeClient, Scheme: scheme, MaxDeployments: 1}
	ctx := context.Background()

	requeue, _, err := r.checkDeploymentCapacity(ctx, current)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if requeue {
		t.Error("expected no requeue when AwaitingCode deployments don't count")
	}
}

// ---------------------------------------------------------------------------
// failureType.String tests
// ---------------------------------------------------------------------------

func TestFailureTypeString(t *testing.T) {
	tests := []struct {
		ft   failureType
		want string
	}{
		{failureApp, "app"},
		{failureInfra, "infra"},
		{failureUnknown, "unknown"},
	}
	for _, tt := range tests {
		if got := tt.ft.String(); got != tt.want {
			t.Errorf("failureType(%d).String() = %q, want %q", tt.ft, got, tt.want)
		}
	}
}
