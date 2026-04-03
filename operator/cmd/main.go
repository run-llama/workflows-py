package main

import (
	"crypto/tls"
	"flag"
	"net/http"
	"net/http/pprof"
	"os"
	"path/filepath"
	"strconv"

	// Import all Kubernetes client auth plugins (e.g. Azure, GCP, OIDC, etc.)
	// to ensure that exec-entrypoint and run can make use of them.
	_ "k8s.io/client-go/plugin/pkg/client/auth"

	"k8s.io/apimachinery/pkg/runtime"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	clientgoscheme "k8s.io/client-go/kubernetes/scheme"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/cache"
	"sigs.k8s.io/controller-runtime/pkg/certwatcher"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/healthz"
	"sigs.k8s.io/controller-runtime/pkg/log/zap"
	ctrlmetrics "sigs.k8s.io/controller-runtime/pkg/metrics"
	"sigs.k8s.io/controller-runtime/pkg/metrics/filters"
	metricsserver "sigs.k8s.io/controller-runtime/pkg/metrics/server"
	"sigs.k8s.io/controller-runtime/pkg/webhook"

	llamadeployv1 "llama-agents-operator/api/v1"
	"llama-agents-operator/internal/controller"
	// +kubebuilder:scaffold:imports
)

var (
	scheme   = runtime.NewScheme()
	setupLog = ctrl.Log.WithName("setup")
)

type Config struct {
	MetricsAddr          string
	ProbeAddr            string
	EnableLeaderElection bool
	SecureMetrics        bool
	EnableHTTP2          bool

	// Webhook certificate options
	WebhookCertPath string
	WebhookCertName string
	WebhookCertKey  string

	// Metrics certificate options
	MetricsCertPath string
	MetricsCertName string
	MetricsCertKey  string
}

func init() {
	utilruntime.Must(clientgoscheme.AddToScheme(scheme))

	utilruntime.Must(llamadeployv1.AddToScheme(scheme))
	// +kubebuilder:scaffold:scheme
}

// nolint:gocyclo
func main() {
	var config Config
	var tlsOpts []func(*tls.Config)

	// Parse flags using standard flag package
	flag.StringVar(&config.MetricsAddr, "metrics-bind-address", "0", "The address the metrics endpoint binds to. "+
		"Use :8443 for HTTPS or :8080 for HTTP, or leave as 0 to disable the metrics service.")
	flag.StringVar(&config.ProbeAddr, "health-probe-bind-address", ":8081", "The address the probe endpoint binds to.")
	flag.BoolVar(&config.EnableLeaderElection, "leader-elect", false,
		"Enable leader election for controller manager. "+
			"Enabling this will ensure there is only one active controller manager.")
	flag.BoolVar(&config.SecureMetrics, "metrics-secure", true,
		"If set, the metrics endpoint is served securely via HTTPS. Use --metrics-secure=false to use HTTP instead.")
	flag.StringVar(&config.WebhookCertPath, "webhook-cert-path", "",
		"The directory that contains the webhook certificate.")
	flag.StringVar(&config.WebhookCertName, "webhook-cert-name", "tls.crt", "The name of the webhook certificate file.")
	flag.StringVar(&config.WebhookCertKey, "webhook-cert-key", "tls.key", "The name of the webhook key file.")
	flag.StringVar(&config.MetricsCertPath, "metrics-cert-path", "",
		"The directory that contains the metrics server certificate.")
	flag.StringVar(&config.MetricsCertName, "metrics-cert-name", "tls.crt",
		"The name of the metrics server certificate file.")
	flag.StringVar(&config.MetricsCertKey, "metrics-cert-key", "tls.key", "The name of the metrics server key file.")
	flag.BoolVar(&config.EnableHTTP2, "enable-http2", false,
		"If set, HTTP/2 will be enabled for the metrics and webhook servers")

	opts := zap.Options{
		Development: false,
	}
	opts.BindFlags(flag.CommandLine)
	flag.Parse()

	ctrl.SetLogger(zap.New(zap.UseFlagOptions(&opts)))

	// if the enable-http2 flag is false (the default), http/2 should be disabled
	// due to its vulnerabilities. More specifically, disabling http/2 will
	// prevent from being vulnerable to the HTTP/2 Stream Cancellation and
	// Rapid Reset CVEs. For more information see:
	// - https://github.com/advisories/GHSA-qppj-fm5r-hxr3
	// - https://github.com/advisories/GHSA-4374-p667-p6c8
	disableHTTP2 := func(c *tls.Config) {
		setupLog.Info("disabling http/2")
		c.NextProtos = []string{"http/1.1"}
	}

	if !config.EnableHTTP2 {
		tlsOpts = append(tlsOpts, disableHTTP2)
	}

	// Create watchers for metrics and webhooks certificates
	var metricsCertWatcher, webhookCertWatcher *certwatcher.CertWatcher

	// Initial webhook TLS options
	webhookTLSOpts := tlsOpts

	if len(config.WebhookCertPath) > 0 {
		setupLog.Info("Initializing webhook certificate watcher using provided certificates",
			"webhook-cert-path", config.WebhookCertPath,
			"webhook-cert-name", config.WebhookCertName,
			"webhook-cert-key", config.WebhookCertKey)

		var err error
		webhookCertWatcher, err = certwatcher.New(
			filepath.Join(config.WebhookCertPath, config.WebhookCertName),
			filepath.Join(config.WebhookCertPath, config.WebhookCertKey),
		)
		if err != nil {
			setupLog.Error(err, "Failed to initialize webhook certificate watcher")
			os.Exit(1)
		}

		webhookTLSOpts = append(webhookTLSOpts, func(config *tls.Config) {
			config.GetCertificate = webhookCertWatcher.GetCertificate
		})
	}

	webhookServer := webhook.NewServer(webhook.Options{
		TLSOpts: webhookTLSOpts,
	})

	// Metrics endpoint is enabled in 'config/default/kustomization.yaml'. The Metrics options configure the server.
	// More info:
	// - https://pkg.go.dev/sigs.k8s.io/controller-runtime@v0.21.0/pkg/metrics/server
	// - https://book.kubebuilder.io/reference/metrics.html
	metricsServerOptions := metricsserver.Options{
		BindAddress:   config.MetricsAddr,
		SecureServing: config.SecureMetrics,
		TLSOpts:       tlsOpts,
	}

	if config.SecureMetrics {
		// FilterProvider is used to protect the metrics endpoint with authn/authz.
		// These configurations ensure that only authorized users and service accounts
		// can access the metrics endpoint. The RBAC are configured in 'config/rbac/kustomization.yaml'. More info:
		// https://pkg.go.dev/sigs.k8s.io/controller-runtime@v0.21.0/pkg/metrics/filters#WithAuthenticationAndAuthorization
		metricsServerOptions.FilterProvider = filters.WithAuthenticationAndAuthorization
	}

	// If the certificate is not specified, controller-runtime will automatically
	// generate self-signed certificates for the metrics server. While convenient for development and testing,
	// this setup is not recommended for production.
	//
	// TODO(user): If you enable certManager, uncomment the following lines:
	// - [METRICS-WITH-CERTS] at config/default/kustomization.yaml to generate and use certificates
	// managed by cert-manager for the metrics server.
	// - [PROMETHEUS-WITH-CERTS] at config/prometheus/kustomization.yaml for TLS certification.
	if len(config.MetricsCertPath) > 0 {
		setupLog.Info("Initializing metrics certificate watcher using provided certificates",
			"metrics-cert-path", config.MetricsCertPath,
			"metrics-cert-name", config.MetricsCertName,
			"metrics-cert-key", config.MetricsCertKey)

		var err error
		metricsCertWatcher, err = certwatcher.New(
			filepath.Join(config.MetricsCertPath, config.MetricsCertName),
			filepath.Join(config.MetricsCertPath, config.MetricsCertKey),
		)
		if err != nil {
			setupLog.Error(err, "to initialize metrics certificate watcher", "error", err)
			os.Exit(1)
		}

		metricsServerOptions.TLSOpts = append(metricsServerOptions.TLSOpts, func(config *tls.Config) {
			config.GetCertificate = metricsCertWatcher.GetCertificate
		})
	}

	// Get the namespace to watch from environment or default to current namespace
	watchNamespace := os.Getenv("WATCH_NAMESPACE")
	if watchNamespace == "" {
		// Try to read the namespace from the service account mount
		if ns, err := os.ReadFile("/var/run/secrets/kubernetes.io/serviceaccount/namespace"); err == nil {
			watchNamespace = string(ns)
		} else {
			watchNamespace = "default"
		}
	}

	mgr, err := ctrl.NewManager(ctrl.GetConfigOrDie(), ctrl.Options{
		Scheme:                 scheme,
		Metrics:                metricsServerOptions,
		WebhookServer:          webhookServer,
		HealthProbeBindAddress: config.ProbeAddr,
		LeaderElection:         config.EnableLeaderElection,
		LeaderElectionID:       "5095c351.deploy.llamaindex.ai",
		// Limit the manager to watch only the specified namespace
		Cache: cache.Options{
			DefaultNamespaces: map[string]cache.Config{
				watchNamespace: {},
			},
		},
		// LeaderElectionReleaseOnCancel defines if the leader should step down voluntarily
		// when the Manager ends. This requires the binary to immediately end when the
		// Manager is stopped, otherwise, this setting is unsafe. Setting this significantly
		// speeds up voluntary leader transitions as the new leader don't have to wait
		// LeaseDuration time first.
		//
		// In the default scaffold provided, the program ends immediately after
		// the manager stops, so would be fine to enable this option. However,
		// if you are doing or is intended to do any operation such as perform cleanups
		// after the manager stops then its usage might be unsafe.
		// LeaderElectionReleaseOnCancel: true,
	})
	if err != nil {
		setupLog.Error(err, "unable to start manager")
		os.Exit(1)
	}

	// Create a non-cached client for operations on types we don't want in the
	// informer cache (e.g. ReplicaSets).  With hundreds of Deployments the RS
	// informer alone can consume gigabytes of memory.
	directClient, err := client.New(ctrl.GetConfigOrDie(), client.Options{Scheme: scheme})
	if err != nil {
		setupLog.Error(err, "unable to create direct API client")
		os.Exit(1)
	}

	// Parse max concurrent rollouts from env var
	maxConcurrentRollouts := 0
	if raw := os.Getenv("LLAMA_DEPLOY_MAX_CONCURRENT_ROLLOUTS"); raw != "" {
		if v, err := strconv.Atoi(raw); err == nil && v >= 0 {
			maxConcurrentRollouts = v
			setupLog.Info("Max concurrent rollouts configured", "limit", v)
		}
	}

	// Parse max deployments limit from env var
	maxDeployments := 0
	if raw := os.Getenv("LLAMA_DEPLOY_MAX_DEPLOYMENTS"); raw != "" {
		if v, err := strconv.Atoi(raw); err == nil && v >= 0 {
			maxDeployments = v
			setupLog.Info("Max deployments configured", "limit", v)
		}
	}

	// Register phase metrics collector
	ctrlmetrics.Registry.MustRegister(controller.NewPhaseCollector(mgr.GetClient()))
	ctrlmetrics.Registry.MustRegister(controller.NewCapacityCollector(mgr.GetClient(), maxDeployments))
	if err := (&controller.LlamaDeploymentReconciler{
		Client:                mgr.GetClient(),
		Scheme:                mgr.GetScheme(),
		Recorder:              mgr.GetEventRecorderFor("llamadeployment-controller"),
		DirectClient:          directClient,
		MaxConcurrentRollouts: maxConcurrentRollouts,
		MaxDeployments:        maxDeployments,
	}).SetupWithManager(mgr); err != nil {
		setupLog.Error(err, "unable to create controller", "controller", "LlamaDeployment")
		os.Exit(1)
	}
	// +kubebuilder:scaffold:builder

	if metricsCertWatcher != nil {
		setupLog.Info("Adding metrics certificate watcher to manager")
		if err := mgr.Add(metricsCertWatcher); err != nil {
			setupLog.Error(err, "unable to add metrics certificate watcher to manager")
			os.Exit(1)
		}
	}

	if webhookCertWatcher != nil {
		setupLog.Info("Adding webhook certificate watcher to manager")
		if err := mgr.Add(webhookCertWatcher); err != nil {
			setupLog.Error(err, "unable to add webhook certificate watcher to manager")
			os.Exit(1)
		}
	}

	if err := mgr.AddHealthzCheck("healthz", healthz.Ping); err != nil {
		setupLog.Error(err, "unable to set up health check")
		os.Exit(1)
	}
	if err := mgr.AddReadyzCheck("readyz", healthz.Ping); err != nil {
		setupLog.Error(err, "unable to set up ready check")
		os.Exit(1)
	}

	// Start pprof server for memory profiling.
	// Usage: kubectl port-forward <pod> 6060:6060
	//   curl -s http://localhost:6060/debug/pprof/heap > heap.prof
	//   go tool pprof heap.prof
	pprofMux := http.NewServeMux()
	pprofMux.HandleFunc("/debug/pprof/", pprof.Index)
	pprofMux.HandleFunc("/debug/pprof/cmdline", pprof.Cmdline)
	pprofMux.HandleFunc("/debug/pprof/profile", pprof.Profile)
	pprofMux.HandleFunc("/debug/pprof/symbol", pprof.Symbol)
	pprofMux.HandleFunc("/debug/pprof/trace", pprof.Trace)
	go func() {
		pprofAddr := ":6060"
		setupLog.Info("starting pprof server", "addr", pprofAddr)
		if err := http.ListenAndServe(pprofAddr, pprofMux); err != nil {
			setupLog.Error(err, "pprof server failed")
		}
	}()

	setupLog.Info("starting manager")
	if err := mgr.Start(ctrl.SetupSignalHandler()); err != nil {
		setupLog.Error(err, "problem running manager")
		os.Exit(1)
	}
}
