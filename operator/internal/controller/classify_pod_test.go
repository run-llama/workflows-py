package controller

import (
	"testing"

	corev1 "k8s.io/api/core/v1"
)

func TestClassifyPod(t *testing.T) {
	tests := []struct {
		name string
		pod  *corev1.Pod
		want failureType
	}{
		{
			name: "evicted pod returns failureInfra",
			pod: &corev1.Pod{
				Status: corev1.PodStatus{
					Reason: "Evicted",
				},
			},
			want: failureInfra,
		},
		{
			name: "pending pod with no container statuses returns failureInfra",
			pod: &corev1.Pod{
				Status: corev1.PodStatus{
					Phase: corev1.PodPending,
				},
			},
			want: failureInfra,
		},
		{
			name: "container in CrashLoopBackOff returns failureApp",
			pod: &corev1.Pod{
				Status: corev1.PodStatus{
					ContainerStatuses: []corev1.ContainerStatus{
						{
							State: corev1.ContainerState{
								Waiting: &corev1.ContainerStateWaiting{
									Reason: "CrashLoopBackOff",
								},
							},
						},
					},
				},
			},
			want: failureApp,
		},
		{
			name: "container in ImagePullBackOff returns failureApp",
			pod: &corev1.Pod{
				Status: corev1.PodStatus{
					ContainerStatuses: []corev1.ContainerStatus{
						{
							State: corev1.ContainerState{
								Waiting: &corev1.ContainerStateWaiting{
									Reason: "ImagePullBackOff",
								},
							},
						},
					},
				},
			},
			want: failureApp,
		},
		{
			name: "container in ErrImagePull returns failureApp",
			pod: &corev1.Pod{
				Status: corev1.PodStatus{
					ContainerStatuses: []corev1.ContainerStatus{
						{
							State: corev1.ContainerState{
								Waiting: &corev1.ContainerStateWaiting{
									Reason: "ErrImagePull",
								},
							},
						},
					},
				},
			},
			want: failureApp,
		},
		{
			name: "container in CreateContainerConfigError returns failureApp",
			pod: &corev1.Pod{
				Status: corev1.PodStatus{
					ContainerStatuses: []corev1.ContainerStatus{
						{
							State: corev1.ContainerState{
								Waiting: &corev1.ContainerStateWaiting{
									Reason: "CreateContainerConfigError",
								},
							},
						},
					},
				},
			},
			want: failureApp,
		},
		{
			name: "container terminated with non-zero exit code returns failureApp",
			pod: &corev1.Pod{
				Status: corev1.PodStatus{
					ContainerStatuses: []corev1.ContainerStatus{
						{
							State: corev1.ContainerState{
								Terminated: &corev1.ContainerStateTerminated{
									ExitCode: 1,
								},
							},
						},
					},
				},
			},
			want: failureApp,
		},
		{
			name: "container OOMKilled with non-zero exit returns failureApp",
			pod: &corev1.Pod{
				Status: corev1.PodStatus{
					ContainerStatuses: []corev1.ContainerStatus{
						{
							State: corev1.ContainerState{
								Terminated: &corev1.ContainerStateTerminated{
									ExitCode: 137,
									Reason:   "OOMKilled",
								},
							},
						},
					},
				},
			},
			want: failureApp,
		},
		{
			name: "init container terminated with non-zero exit code returns failureApp",
			pod: &corev1.Pod{
				Status: corev1.PodStatus{
					InitContainerStatuses: []corev1.ContainerStatus{
						{
							State: corev1.ContainerState{
								Terminated: &corev1.ContainerStateTerminated{
									ExitCode: 1,
								},
							},
						},
					},
				},
			},
			want: failureApp,
		},
		{
			name: "running pod with no issues returns failureUnknown",
			pod: &corev1.Pod{
				Status: corev1.PodStatus{
					Phase: corev1.PodRunning,
					ContainerStatuses: []corev1.ContainerStatus{
						{
							State: corev1.ContainerState{
								Running: &corev1.ContainerStateRunning{},
							},
						},
					},
				},
			},
			want: failureUnknown,
		},
		{
			name: "container with LastTerminationState non-zero exit returns failureApp",
			pod: &corev1.Pod{
				Status: corev1.PodStatus{
					ContainerStatuses: []corev1.ContainerStatus{
						{
							State: corev1.ContainerState{
								Running: &corev1.ContainerStateRunning{},
							},
							LastTerminationState: corev1.ContainerState{
								Terminated: &corev1.ContainerStateTerminated{
									ExitCode: 1,
								},
							},
						},
					},
				},
			},
			want: failureApp,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := classifyPod(tt.pod)
			if got != tt.want {
				t.Errorf("classifyPod() = %v (%d), want %v (%d)", got, got, tt.want, tt.want)
			}
		})
	}
}
