from layer_naive import MultiLayer

apple_price = 100
apple_unit = 2
tax = 1.1

# layer
mul_apple_layer = MultiLayer()
mul_tax_layer = MultiLayer()

## forward
apple_cost= mul_apple_layer.forward(apple_price,apple_unit)
amount = mul_tax_layer.forward(apple_cost,tax)

## 总金额
print(amount)

## backward
d_amount = 1
d_apple_cost,d_tax = mul_tax_layer.backward(d_amount)
d_apple_price,d_apple_unit = mul_apple_layer.backward(d_apple_cost)

print(d_apple_price,d_apple_unit,d_tax)

