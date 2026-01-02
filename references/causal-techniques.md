# Causal Analysis Techniques Reference

Methods for establishing causal relationships in systems.

## Tracer Injection

### Numerical Tracers
```python
MAGIC_NUMBERS = {
    'hex': [0xDEADBEEF, 0xCAFEBABE, 0xC0FFEE00, 0x8BADF00D],
    'prime': [982451653, 32416190071, 479001599],
    'float': [3.141592653589793, 2.718281828459045, 1.618033988749895]
}

def inject_tracer(system, channel, tracer_type='hex'):
    """Inject tracer and observe propagation."""
    tracer = random.choice(MAGIC_NUMBERS[tracer_type])
    
    # Record injection point
    injection_time = time.time()
    system.inject(channel, tracer)
    
    # Monitor for tracer emergence
    observations = []
    timeout = 10.0  # seconds
    
    while time.time() - injection_time < timeout:
        for output_channel in system.outputs:
            value = system.observe(output_channel)
            if contains_tracer(value, tracer):
                observations.append({
                    'channel': output_channel,
                    'latency': time.time() - injection_time,
                    'transformation': identify_transform(tracer, value)
                })
    
    return observations
```

### String Tracers
```python
def generate_string_tracer():
    """Generate unique string tracer."""
    import uuid
    return f"__PROBE_{uuid.uuid4().hex[:8]}__"

def trace_string_propagation(system, tracer):
    """Track string tracer through system."""
    paths = []
    
    # Inject at all input points
    for input_ch in system.inputs:
        system.inject(input_ch, tracer)
    
    # Search for tracer in outputs, logs, files
    search_locations = system.outputs + system.log_files + system.temp_files
    
    for loc in search_locations:
        content = read_location(loc)
        if tracer in content:
            # Found! Extract context
            idx = content.find(tracer)
            context = content[max(0,idx-50):idx+len(tracer)+50]
            paths.append({
                'location': loc,
                'context': context,
                'transformed': tracer != extract_tracer(content)
            })
    
    return paths
```

## Differential Analysis

### Single Variable Perturbation
```python
def differential_analysis(system, variable, delta, n_trials=10):
    """
    Measure causal effect of variable by perturbation.
    
    Returns: {output: (effect_size, p_value)} for each output
    """
    # Baseline measurements
    baseline = []
    for _ in range(n_trials):
        baseline.append(system.measure_outputs())
    
    # Perturbed measurements
    perturbed = []
    for _ in range(n_trials):
        system.set(variable, system.get(variable) + delta)
        perturbed.append(system.measure_outputs())
        system.reset(variable)
    
    # Statistical comparison
    results = {}
    for output in baseline[0].keys():
        b_values = [m[output] for m in baseline]
        p_values = [m[output] for m in perturbed]
        
        # Effect size
        effect = np.mean(p_values) - np.mean(b_values)
        
        # Statistical test (t-test)
        from scipy.stats import ttest_ind
        _, p_value = ttest_ind(b_values, p_values)
        
        results[output] = {
            'effect_size': effect,
            'relative_effect': effect / (np.mean(b_values) + 1e-10),
            'p_value': p_value,
            'significant': p_value < 0.05
        }
    
    return results
```

### Factorial Design
```python
def factorial_analysis(system, factors, levels=2):
    """
    Full factorial experiment to find main effects and interactions.
    
    factors: list of factor names
    levels: number of levels per factor (default 2: low/high)
    """
    from itertools import product
    
    # Generate all combinations
    factor_levels = {f: list(range(levels)) for f in factors}
    combinations = list(product(*[range(levels) for _ in factors]))
    
    results = []
    for combo in combinations:
        # Set factor levels
        config = dict(zip(factors, combo))
        system.configure(config)
        
        # Measure
        output = system.measure()
        results.append({'config': config, 'output': output})
    
    # Compute main effects
    main_effects = {}
    for factor in factors:
        high = [r['output'] for r in results if r['config'][factor] == 1]
        low = [r['output'] for r in results if r['config'][factor] == 0]
        main_effects[factor] = np.mean(high) - np.mean(low)
    
    # Compute 2-way interactions
    interactions = {}
    for i, f1 in enumerate(factors):
        for f2 in factors[i+1:]:
            # Interaction = difference in effect of f1 at different f2 levels
            f1_effect_at_f2_high = (
                np.mean([r['output'] for r in results 
                        if r['config'][f1]==1 and r['config'][f2]==1]) -
                np.mean([r['output'] for r in results 
                        if r['config'][f1]==0 and r['config'][f2]==1])
            )
            f1_effect_at_f2_low = (
                np.mean([r['output'] for r in results 
                        if r['config'][f1]==1 and r['config'][f2]==0]) -
                np.mean([r['output'] for r in results 
                        if r['config'][f1]==0 and r['config'][f2]==0])
            )
            interactions[f"{f1}×{f2}"] = f1_effect_at_f2_high - f1_effect_at_f2_low
    
    return {'main_effects': main_effects, 'interactions': interactions}
```

## Causal Graph Construction

### Graph Data Structure
```python
from dataclasses import dataclass, field
from typing import List, Dict, Optional

@dataclass
class CausalNode:
    id: str
    name: str
    type: str  # 'variable', 'component', 'process'
    attributes: Dict = field(default_factory=dict)

@dataclass  
class CausalEdge:
    source: str
    target: str
    weight: float  # Strength of causal effect
    delay: float   # Time delay (0 if instantaneous)
    type: str      # 'direct', 'mediated', 'feedback'
    confidence: float  # 0-1

class CausalGraph:
    def __init__(self):
        self.nodes: Dict[str, CausalNode] = {}
        self.edges: List[CausalEdge] = []
    
    def add_node(self, node: CausalNode):
        self.nodes[node.id] = node
    
    def add_edge(self, edge: CausalEdge):
        self.edges.append(edge)
    
    def get_causes(self, node_id: str) -> List[str]:
        """Get direct causes of a node."""
        return [e.source for e in self.edges if e.target == node_id]
    
    def get_effects(self, node_id: str) -> List[str]:
        """Get direct effects of a node."""
        return [e.target for e in self.edges if e.source == node_id]
    
    def find_feedback_loops(self) -> List[List[str]]:
        """Find all feedback loops in graph."""
        loops = []
        
        def dfs(start, current, path, visited):
            if current == start and len(path) > 1:
                loops.append(path.copy())
                return
            if current in visited:
                return
            
            visited.add(current)
            for effect in self.get_effects(current):
                dfs(start, effect, path + [effect], visited)
            visited.remove(current)
        
        for node_id in self.nodes:
            dfs(node_id, node_id, [node_id], set())
        
        return loops
    
    def classify_loops(self) -> Dict[str, str]:
        """Classify feedback loops as reinforcing (R) or balancing (B)."""
        loops = self.find_feedback_loops()
        classifications = {}
        
        for loop in loops:
            # Count negative effects in loop
            negative_count = 0
            for i in range(len(loop)):
                src, tgt = loop[i], loop[(i+1) % len(loop)]
                edge = next(e for e in self.edges if e.source == src and e.target == tgt)
                if edge.weight < 0:
                    negative_count += 1
            
            # Odd number of negatives = balancing, even = reinforcing
            loop_id = "→".join(loop)
            classifications[loop_id] = 'B' if negative_count % 2 == 1 else 'R'
        
        return classifications
    
    def to_dot(self) -> str:
        """Export to GraphViz DOT format."""
        lines = ["digraph CausalGraph {"]
        
        for node in self.nodes.values():
            lines.append(f'  "{node.id}" [label="{node.name}"];')
        
        for edge in self.edges:
            style = "dashed" if edge.delay > 0 else "solid"
            color = "red" if edge.weight < 0 else "black"
            lines.append(
                f'  "{edge.source}" -> "{edge.target}" '
                f'[style={style}, color={color}, label="{edge.weight:.2f}"];'
            )
        
        lines.append("}")
        return "\n".join(lines)
```

## Falsification Protocol

### Hypothesis Test Design
```python
def design_falsification_test(hypothesis):
    """
    Design a test that maximally discriminates hypothesis from alternatives.
    
    Key principle: Design test most likely to REFUTE hypothesis if false.
    """
    # Extract predictions from hypothesis
    predictions = hypothesis.predict_behavior()
    
    # Find prediction with narrowest confidence interval
    # (most specific prediction = most falsifiable)
    most_specific = min(predictions, key=lambda p: p.confidence_interval_width)
    
    # Design test targeting that prediction
    test = {
        'target_prediction': most_specific,
        'input_conditions': most_specific.required_inputs,
        'expected_output': most_specific.expected_value,
        'tolerance': most_specific.confidence_interval,
        'falsification_criterion': 'output outside tolerance'
    }
    
    return test

def execute_falsification_test(system, test, n_trials=10):
    """Execute falsification test and assess result."""
    results = []
    
    for _ in range(n_trials):
        system.configure(test['input_conditions'])
        output = system.measure()
        
        in_tolerance = (
            test['expected_output'] - test['tolerance'] <= output <=
            test['expected_output'] + test['tolerance']
        )
        results.append({'output': output, 'in_tolerance': in_tolerance})
    
    # Assess
    pass_rate = sum(r['in_tolerance'] for r in results) / n_trials
    
    if pass_rate < 0.5:
        verdict = 'REFUTED'
    elif pass_rate < 0.9:
        verdict = 'WEAKENED'
    else:
        verdict = 'CORROBORATED'
    
    return {
        'verdict': verdict,
        'pass_rate': pass_rate,
        'results': results
    }
```

## Dependency Matrix

### Construction
```python
def build_dependency_matrix(graph: CausalGraph) -> np.ndarray:
    """Build adjacency matrix from causal graph."""
    nodes = list(graph.nodes.keys())
    n = len(nodes)
    node_idx = {node: i for i, node in enumerate(nodes)}
    
    matrix = np.zeros((n, n))
    
    for edge in graph.edges:
        i = node_idx[edge.source]
        j = node_idx[edge.target]
        matrix[i, j] = edge.weight
    
    return matrix, nodes

def transitive_closure(matrix: np.ndarray) -> np.ndarray:
    """Compute transitive closure (indirect dependencies)."""
    n = matrix.shape[0]
    closure = matrix.copy()
    
    # Floyd-Warshall variant
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if closure[i, k] != 0 and closure[k, j] != 0:
                    closure[i, j] = 1  # Indirect path exists
    
    return closure
```
