package v1

import (
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// LlamaDeploymentTemplateSpec defines the desired overlay for a LlamaDeployment's PodTemplate.
// This is intended to carry scheduling-related fields like node selectors, tolerations, and
// affinity, but supports any partial PodTemplateSpec. Fields set here will take precedence
// over the operator-computed defaults when merged.
type LlamaDeploymentTemplateSpec struct {
	// PodSpec holds a partial PodTemplateSpec to be merged into the generated PodTemplate.
	// +optional
	// +kubebuilder:pruning:PreserveUnknownFields
	// +kubebuilder:validation:Schemaless
	PodSpec corev1.PodTemplateSpec `json:"podSpec,omitempty"`
}

// +kubebuilder:object:root=true
// +kubebuilder:subresource:status
// +kubebuilder:printcolumn:name="Age",type=date,JSONPath=`.metadata.creationTimestamp`

// LlamaDeploymentTemplate configures default Pod template fields for LlamaDeployments.
// The resource name is referenced by LlamaDeployment.spec.templateName. A special name
// "default" is used as a fallback when no templateName is provided.
type LlamaDeploymentTemplate struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	Spec   LlamaDeploymentTemplateSpec `json:"spec,omitempty"`
	Status metav1.ConditionStatus      `json:"status,omitempty"`
}

// +kubebuilder:object:root=true

// LlamaDeploymentTemplateList contains a list of LlamaDeploymentTemplate.
type LlamaDeploymentTemplateList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []LlamaDeploymentTemplate `json:"items"`
}

func init() {
	SchemeBuilder.Register(&LlamaDeploymentTemplate{}, &LlamaDeploymentTemplateList{})
}
