layer = int(input())
numblocks = int(input())

results = []


# print("subgraph cluster_{}_0 \{".format(layer))

print('\n')
print(f'label="[layer{layer}_0]";')
print(f'layer{layer}_0_relu0 [label="relu"];')
print(f'layer{layer}_0_relu1 [label="relu"];')
print(f'layer{layer}_0_relu2 [label="relu"];')
print(f'layer{layer}_0_add [label="+"];')

print(f'layer{layer}_0_conv1 [label="conv1"];')
print(f'layer{layer}_0_conv2 [label="conv2"];')
print(f'layer{layer}_0_conv3 [label="conv3"];')
print(f'layer{layer}_0_bn1 [label="bn1"];')
print(f'layer{layer}_0_bn2 [label="bn2"];')
print(f'layer{layer}_0_bn3 [label="bn3"];')
print(f'layer{layer}_0_downsample_relu [label="downsample_relu"];')
print(f'layer{layer}_0_downsample_bn [label="downsample_bn"];')


print(f'layer{layer}_0_conv1 -> layer{layer}_0_bn1;')
print(f'layer{layer}_0_bn1 -> layer{layer}_0_relu0;')
print(f'layer{layer}_0_relu0 -> layer{layer}_0_conv2;')

print(f'layer{layer}_0_conv2 -> layer{layer}_0_bn2;')
print(f'layer{layer}_0_bn2 -> layer{layer}_0_relu1;')
print(f'layer{layer}_0_relu1 -> layer{layer}_0_conv3;')

print(f'layer{layer}_0_conv3 -> layer{layer}_0_bn3;')

print(f'layer{layer}_0_downsample_relu -> layer{layer}_0_downsample_bn;')

print(f'layer{layer}_0_bn3 -> layer{layer}_0_add;')
print(f'layer{layer}_0_downsample_bn -> layer{layer}_0_add;')
print(f'layer{layer}_0_add -> layer{layer}_0_relu2;')

print()

for i in range(numblocks-1):
    idx = i+1
    print(f'label="[layer{layer}_{idx}]";')

    print(f'layer{layer}_{idx}_relu1 [label="[relu]"];')
    print(f'layer{layer}_{idx}_relu2 [label="[relu]"];')
    print(f'layer{layer}_{idx}_relu3 [label="[relu]"];')
    print(f'layer{layer}_{idx}_add [label="+"];')

    print(f'layer{layer}_{idx}_conv1 [label="[conv1]"];')
    print(f'layer{layer}_{idx}_conv2 [label="[conv2]"];')
    print(f'layer{layer}_{idx}_conv3 [label="[conv3]"];')

    print(f'layer{layer}_{idx}_bn1 [label="[bn1]"];')
    print(f'layer{layer}_{idx}_bn2 [label="[bn2]"];')
    print(f'layer{layer}_{idx}_bn3 [label="[bn3]"];')

    print(f'layer{layer}_{idx}_conv1 -> layer{layer}_{idx}_bn1;')
    print(f'layer{layer}_{idx}_bn1 -> layer{layer}_{idx}_relu1;')

    print(f'layer{layer}_{idx}_relu1 -> layer{layer}_{idx}_conv2;')
    print(f'layer{layer}_{idx}_conv2 -> layer{layer}_{idx}_bn2;')
    print(f'layer{layer}_{idx}_bn2 -> layer{layer}_{idx}_relu2;')

    print(f'layer{layer}_{idx}_relu2 -> layer{layer}_{idx}_conv3;')
    print(f'layer{layer}_{idx}_conv3 -> layer{layer}_{idx}_bn3')
    print(f'layer{layer}_{idx}_bn3 -> layer{layer}_{idx}_add;')

    print(f'layer{layer}_{idx}_add -> layer{layer}_{idx}_relu3;')
    print()


print(f'layer{layer}_{0}_relu2 -> layer{layer}_{1}_conv1;')
print(f'layer{layer}_{0}_relu2 -> layer{layer}_{1}_add;')

for i in range(numblocks-1):
    idx = i+1
    print(f'layer{layer}_{idx}_relu3 -> layer{layer}_{idx+1}_conv1;')
    print(f'layer{layer}_{idx}_relu3 -> layer{layer}_{idx+1}_add;')


# layer2_0_relu2 -> layer2_1_conv1;
# layer2_0_relu2 -> layer2_1_add;

# layer2_1_relu3 -> layer2_2_conv1;
# layer2_1_relu3 -> layer2_2_add;

# layer2_2_relu3 -> layer2_3_conv1;
# layer2_2_relu3 -> layer2_3_add;