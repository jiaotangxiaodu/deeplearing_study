from layer_naive import MultiLayer
from layer_naive import AddLayer

apple_price = 100
apple_unit = 2
orange_price = 150
orange_unit = 3
tax = 1.1

mul_apple_cost_layer = MultiLayer()
mul_orange_cost_layer = MultiLayer()
mul_cost_layer = AddLayer()
mul_amount_layer = MultiLayer()

apple_cost = mul_apple_cost_layer.forward(apple_price,apple_unit)
orange_cost = mul_orange_cost_layer.forward(orange_price,orange_unit)
cost = mul_cost_layer.forward(apple_cost,orange_cost)
amount = mul_amount_layer.forward(cost,tax)

print(amount)

d_amount = 1
d_cost,d_tax = mul_amount_layer.backward(d_amount)
d_apple_cost,d_orange_cost = mul_cost_layer.backward(d_cost)
d_apple_price,d_apple_unit = mul_apple_cost_layer.backward(d_apple_cost)
d_orange_price,d_orange_unit = mul_orange_cost_layer.backward(d_orange_cost)
print(d_apple_price,d_apple_unit,d_orange_price,d_orange_unit,d_tax)