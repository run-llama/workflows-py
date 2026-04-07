//go:build !integration

package controller

import (
	"os"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	llamadeployv1 "llama-agents-operator/api/v1"
)

func TestGetContainerImageTag_Precedence_SpecOverridesEnv(t *testing.T) {
	prev := os.Getenv(EnvImageTag)
	t.Cleanup(func() { _ = os.Setenv(EnvImageTag, prev) })

	const specTag = "spec-tag"

	// When both spec and env are set, spec should win (per-deployment pinning)
	_ = os.Setenv(EnvImageTag, "env-tag")
	ld := &llamadeployv1.LlamaDeployment{ObjectMeta: metav1.ObjectMeta{Name: "demo"}}
	ld.Spec.ImageTag = specTag
	if got := getContainerImageTag(ld); got != specTag {
		t.Fatalf("expected spec tag to win, got %q", got)
	}

	// When spec is set and env is empty, spec should be used
	_ = os.Unsetenv(EnvImageTag)
	ld.Spec.ImageTag = "spec-only"
	if got := getContainerImageTag(ld); got != "spec-only" {
		t.Fatalf("expected spec tag when env unset, got %q", got)
	}

	// When env is set and spec empty, env should be used (fallback)
	_ = os.Setenv(EnvImageTag, "env-only")
	ld.Spec.ImageTag = ""
	if got := getContainerImageTag(ld); got != "env-only" {
		t.Fatalf("expected env tag when spec empty, got %q", got)
	}

	// When both unset, default should be used
	_ = os.Unsetenv(EnvImageTag)
	ld.Spec.ImageTag = ""
	if got := getContainerImageTag(ld); got != DefaultImageTag {
		t.Fatalf("expected default tag %q, got %q", DefaultImageTag, got)
	}

	// Legacy "appserver-" prefix should be stripped from spec tag
	ld.Spec.ImageTag = "appserver-0.7.2"
	if got := getContainerImageTag(ld); got != "0.7.2" {
		t.Fatalf("expected appserver- prefix to be stripped, got %q", got)
	}

	// Plain version spec tag should pass through unchanged
	ld.Spec.ImageTag = "0.9.3"
	if got := getContainerImageTag(ld); got != "0.9.3" {
		t.Fatalf("expected plain version tag, got %q", got)
	}
}
