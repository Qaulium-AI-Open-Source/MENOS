import { defineConfig } from 'vitepress'

// Sidebar, grouped to follow how a reader learns MENOS: what it is, how to get
// started, how to use it, how it works, and then the reference material.
// scripts/check-nav.mjs fails the build if a page is missing from this list or an
// entry here has no page.
const sidebar = [
  {
    text: 'Introduction',
    collapsed: false,
    items: [
      { text: 'Overview', link: '/introduction/overview' },
      { text: 'Why MENOS', link: '/introduction/why-menos' },
      { text: 'Architecture', link: '/introduction/architecture' },
      { text: 'Core Concepts', link: '/introduction/core-concepts' },
      { text: 'Supported Platforms', link: '/introduction/supported-platforms' },
    ],
  },
  {
    text: 'Get Started',
    collapsed: false,
    items: [
      { text: 'Installation', link: '/get-started/installation' },
      { text: 'Requirements', link: '/get-started/requirements' },
      { text: 'Verify Installation', link: '/get-started/verify-installation' },
      { text: 'Your First Circuit', link: '/get-started/first-circuit' },
      { text: 'Your First Simulation', link: '/get-started/first-simulation' },
      { text: 'Qiskit Integration', link: '/get-started/qiskit-integration' },
    ],
  },
  {
    text: 'Build with MENOS',
    collapsed: true,
    items: [
      { text: 'Circuits and Gates', link: '/build/circuits-and-gates' },
      { text: 'MENOSBackend', link: '/build/menos-backend' },
      { text: 'Running Jobs', link: '/build/running-jobs' },
      { text: 'Results', link: '/build/results' },
      { text: 'Counts', link: '/build/counts' },
      { text: 'Statevectors', link: '/build/statevectors' },
      { text: 'Expectation Values', link: '/build/expectation-values' },
      { text: 'EstimatorV2', link: '/build/estimator' },
      { text: 'SamplerV2', link: '/build/sampler' },
      { text: 'Noise Models', link: '/build/noise-models' },
      { text: 'Readout Mitigation', link: '/build/readout-mitigation' },
      { text: 'PennyLane Device', link: '/build/pennylane' },
      { text: 'Native API', link: '/build/native-api' },
      { text: 'Command Line', link: '/build/command-line' },
    ],
  },
  {
    text: 'Execution',
    collapsed: true,
    items: [
      { text: 'How MENOS Executes', link: '/execution/how-menos-executes' },
      { text: 'Analyze', link: '/execution/analyze' },
      { text: 'Estimate', link: '/execution/estimate' },
      { text: 'Plan', link: '/execution/plan' },
      { text: 'Execute', link: '/execution/execute' },
      { text: 'Monitor', link: '/runtime/monitoring' },
      { text: 'Replan', link: '/runtime/replanning' },
    ],
  },
  {
    text: 'Simulation Engines',
    collapsed: true,
    items: [
      { text: 'Dense', link: '/engines/dense' },
      { text: 'CUDA', link: '/engines/cuda' },
      { text: 'Stabilizer', link: '/engines/stabilizer' },
      { text: 'Pauli Propagation', link: '/engines/pauli' },
      { text: 'MPS', link: '/engines/mps' },
      { text: 'Tensor Network', link: '/engines/tensor-network' },
      { text: 'Cut and Recombine', link: '/engines/cut-and-recombine' },
      { text: 'Density Matrix', link: '/engines/density-matrix' },
      { text: 'Trajectory', link: '/engines/trajectory' },
    ],
  },
  {
    text: 'Planning',
    collapsed: true,
    items: [
      { text: 'Cost Model', link: '/planning/cost-model' },
      { text: 'Execution Plans', link: '/planning/execution-plans' },
      { text: 'Backend Selection', link: '/planning/backend-selection' },
      { text: 'Memory Constraints', link: '/planning/memory-constraints' },
      { text: 'Plan Explanation', link: '/planning/explain' },
    ],
  },
  {
    text: 'Observables',
    collapsed: true,
    items: [
      { text: 'Pauli Observables', link: '/observables/pauli-observables' },
      { text: 'SparsePauliOp', link: '/observables/sparse-pauli-op' },
      { text: 'Light-Cone Reduction', link: '/observables/light-cone-reduction' },
    ],
  },
  {
    text: 'GPU',
    collapsed: true,
    items: [
      { text: 'CUDA Installation', link: '/gpu/cuda-installation' },
      { text: 'CUDA Requirements', link: '/gpu/cuda-requirements' },
      { text: 'CPU vs CUDA', link: '/gpu/cpu-vs-cuda' },
      { text: 'GPU Memory', link: '/gpu/gpu-memory' },
      { text: 'Troubleshooting CUDA', link: '/gpu/troubleshooting-cuda' },
    ],
  },
  {
    text: 'Runtime',
    collapsed: true,
    items: [
      { text: 'Monitoring', link: '/runtime/monitoring' },
      { text: 'Time Budgets', link: '/runtime/time-budgets' },
      { text: 'Replanning', link: '/runtime/replanning' },
      { text: 'ResourceExceeded', link: '/runtime/resource-exceeded' },
    ],
  },
  {
    text: 'Concepts',
    collapsed: true,
    items: [
      { text: 'KIR', link: '/concepts/kir' },
      { text: 'Conventions', link: '/concepts/conventions' },
      { text: 'Determinism', link: '/concepts/determinism' },
      { text: 'Exactness', link: '/concepts/exactness' },
    ],
  },
  {
    text: 'Examples',
    collapsed: true,
    items: [
      { text: 'All Examples', link: '/examples/' },
      { text: 'Bell State', link: '/examples/bell-state' },
      { text: 'GHZ State', link: '/examples/ghz-state' },
      { text: 'Counts', link: '/examples/counts' },
      { text: 'Statevector', link: '/examples/statevector' },
      { text: 'Large Statevector', link: '/examples/large-statevector' },
      { text: 'Random Circuits', link: '/examples/random-circuits' },
      { text: 'Expectation Value', link: '/examples/expectation-value' },
      { text: 'SparsePauliOp', link: '/examples/sparse-pauli-op' },
      { text: 'EstimatorV2', link: '/examples/estimator' },
      { text: 'Planner Explanation', link: '/examples/planner-explain' },
      { text: 'Stabilizer', link: '/examples/stabilizer' },
      { text: 'MPS', link: '/examples/mps' },
      { text: 'Circuit Cutting', link: '/examples/cut-and-recombine' },
      { text: 'CUDA', link: '/examples/cuda' },
      { text: 'Time Budget', link: '/examples/time-budget' },
      { text: 'Replanning', link: '/examples/replanning' },
      { text: 'Noise', link: '/examples/noise' },
      { text: 'PennyLane', link: '/examples/pennylane' },
    ],
  },
  {
    text: 'API Reference',
    collapsed: true,
    items: [
      { text: 'Python API', link: '/reference/python-api' },
      { text: 'Qiskit Backend', link: '/reference/qiskit-backend' },
      { text: 'Estimator', link: '/reference/estimator-api' },
      { text: 'Job', link: '/reference/job-api' },
      { text: 'Results', link: '/reference/results-api' },
      { text: 'Configuration', link: '/reference/configuration' },
      { text: 'Exceptions', link: '/reference/exceptions' },
    ],
  },
  {
    text: 'Reference',
    collapsed: true,
    items: [
      { text: 'Supported Gates', link: '/reference/supported-gates' },
      { text: 'Supported Observables', link: '/reference/supported-observables' },
      { text: 'Compatibility', link: '/reference/compatibility' },
      { text: 'Limitations', link: '/reference/limitations' },
      { text: 'Changelog', link: '/reference/changelog' },
    ],
  },
  {
    text: 'Troubleshooting',
    collapsed: true,
    items: [
      { text: 'Installation Problems', link: '/troubleshooting/installation' },
      { text: 'Build Errors', link: '/troubleshooting/build-errors' },
      { text: 'CUDA Errors', link: '/troubleshooting/cuda-errors' },
      { text: 'Memory Errors', link: '/troubleshooting/memory-errors' },
      { text: 'Unsupported Circuits', link: '/troubleshooting/unsupported-circuits' },
      { text: 'Common Errors', link: '/troubleshooting/common-errors' },
    ],
  },
]

export default defineConfig({
  title: 'MENOS',
  titleTemplate: ':title | MENOS Docs',
  description:
    'MENOS is a quantum circuit simulation compiler and runtime. It analyzes a circuit, estimates candidate execution strategies, chooses a plan, and runs it.',
  lang: 'en-US',
  appearance: 'force-dark',
  cleanUrls: true,
  // Only the public documentation is built. Architecture decision records and the
  // internal build plan, benchmark records and engineering notes stay in the
  // repository, unpublished by the site.
  srcExclude: ['adr/**', 'benchmarks/**', 'BUILD_PLAN.md', 'ARCHITECTURE_NOTES.md', 'ENGINE_TECHNIQUES.md', 'node_modules/**'],
  head: [['link', { rel: 'icon', type: 'image/png', href: '/favicon.png' }]],
  themeConfig: {
    logo: { src: '/logo.png', alt: 'MENOS' },
    siteTitle: false,
    nav: [
      { text: 'Get Started', link: '/get-started/installation', activeMatch: '/get-started/' },
      { text: 'Guides', link: '/build/menos-backend', activeMatch: '/(build|execution|observables)/' },
      { text: 'Engines', link: '/engines/dense', activeMatch: '/engines/' },
      { text: 'Planning', link: '/planning/cost-model', activeMatch: '/(planning|runtime)/' },
      { text: 'GPU', link: '/gpu/cuda-installation', activeMatch: '/gpu/' },
      { text: 'API Reference', link: '/reference/python-api', activeMatch: '/reference/' },
      { text: 'Examples', link: '/examples/', activeMatch: '/examples/' },
      { text: 'Troubleshooting', link: '/troubleshooting/common-errors', activeMatch: '/troubleshooting/' },
    ],
    sidebar,
    search: { provider: 'local' },
    lastUpdated: { text: 'Last updated' },
    returnToTopLabel: 'Back to top',
    sidebarMenuLabel: 'Menu',
    darkModeSwitchLabel: 'Theme',
    outline: { level: [2, 3], label: 'On this page' },
    docFooter: { prev: 'Previous', next: 'Next' },
    footer: {
      message: 'Released under the Apache-2.0 license.',
      copyright: 'Qaulium AI',
    },
  },
  markdown: {
    theme: { light: 'github-light', dark: 'github-dark' },
  },
})
