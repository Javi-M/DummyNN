mutable struct GenericNeuron{T}
    inputs::Vector{T}
    weights::Vector{T}
    bias::T
    weigh_fn::Function     # This function will be used where `*` is usually used
    aggregate_fn::Function # This function will be used where `+` is usually used
    activation_fn::Function # Function that take the sin
end

"Compute the output value of a neuron"
function forward(neuron::GenericNeuron{T}) where T
    # Weighted inputs
    weighted_inputs = [neuron.weigh_fn(w, x) for (w, x) in zip(neuron.weights, neuron.inputs)]
    
    # Aggregate them with the bias
    synaptic_potential = foldl(neuron.aggregate_fn, weighted_inputs; init=neuron.bias)
    
    # Apply the activation function to the synaptic potential
    return neuron.activation_fn(synaptic_potential)
end


"Calculates the error commited"


# EXAMPLES

# Example 1: simple normal neuron, with 5 outputs and the activation function being the identity
num_inputs = 5
inputs = zeros(Float32, num_inputs)
weights = rand(Float32, num_inputs)
bias = rand(Float32)
simple_neuron = GenericNeuron{Float32}(inputs, weights, bias, (*), (+), x->x)

println(simple_neuron)

println(forward(simple_neuron))