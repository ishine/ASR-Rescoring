import torch 
a = torch.tensor(
    [[[[1,5],[2,3],[3,1]],[[2,4],[3,2],[4,9]]],
    [[[4,1],[5,2],[6,3]],[[5,4],[6,5],[7,6]]]]
)
print(a)
print(a.size())

a = a.reshape(
    (a.size(dim=0), a.size(dim=1), a.size(dim=2)*a.size(dim=3))
)
print(a)
print(a.size())
#print(torch.cat([t for t in a[]]))