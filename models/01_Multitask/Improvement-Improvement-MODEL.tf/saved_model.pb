��8
� � 
D
AddV2
x"T
y"T
z"T"
Ttype:
2	��
h
Any	
input

reduction_indices"Tidx

output
"
	keep_dimsbool( "
Tidxtype0:
2	
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
�
BiasAdd

value"T	
bias"T
output"T""
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
$
DisableCopyOnRead
resource�
R
Einsum
inputs"T*N
output"T"
equationstring"
Nint(0"	
Ttype
*
Erf
x"T
y"T"
Ttype:
2
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
�
FusedBatchNormV3
x"T

scale"U
offset"U	
mean"U
variance"U
y"T

batch_mean"U
batch_variance"U
reserve_space_1"U
reserve_space_2"U
reserve_space_3"U"
Ttype:
2"
Utype:
2"
epsilonfloat%��8"&
exponential_avg_factorfloat%  �?";
data_formatstringNHWC:
NHWCNCHWNDHWCNCDHW"
is_trainingbool(
�
GatherV2
params"Tparams
indices"Tindices
axis"Taxis
output"Tparams"

batch_dimsint "
Tparamstype"
Tindicestype:
2	"
Taxistype:
2	
.
Identity

input"T
output"T"	
Ttype
u
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
2	
�
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( �
?
Mul
x"T
y"T
z"T"
Ttype:
2	�

NoOp
U
NotEqual
x"T
y"T
z
"	
Ttype"$
incompatible_shape_errorbool(�
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
�
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( ""
Ttype:
2	"
Tidxtype0:
2	
@
ReadVariableOp
resource
value"dtype"
dtypetype�
@
RealDiv
x"T
y"T
z"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
d
Shape

input"T&
output"out_type��out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
9
Softmax
logits"T
softmax"T"
Ttype:
2
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ��
@
StaticRegexFullMatch	
input

output
"
patternstring
�
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
L

StringJoin
inputs*N

output"

Nint("
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	
�
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( ""
Ttype:
2	"
Tidxtype0:
2	
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.14.02v2.14.0-rc1-21-g4dacf3f368e8��2
�
7transformer_encoder_5/Encoder-FeedForwardLayer_2_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*H
shared_name97transformer_encoder_5/Encoder-FeedForwardLayer_2_3/bias
�
Ktransformer_encoder_5/Encoder-FeedForwardLayer_2_3/bias/Read/ReadVariableOpReadVariableOp7transformer_encoder_5/Encoder-FeedForwardLayer_2_3/bias*
_output_shapes
:	*
dtype0
�
9transformer_encoder_5/Encoder-FeedForwardLayer_2_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:		*J
shared_name;9transformer_encoder_5/Encoder-FeedForwardLayer_2_3/kernel
�
Mtransformer_encoder_5/Encoder-FeedForwardLayer_2_3/kernel/Read/ReadVariableOpReadVariableOp9transformer_encoder_5/Encoder-FeedForwardLayer_2_3/kernel*
_output_shapes

:		*
dtype0
�
7transformer_encoder_5/Encoder-FeedForwardLayer_1_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*H
shared_name97transformer_encoder_5/Encoder-FeedForwardLayer_1_3/bias
�
Ktransformer_encoder_5/Encoder-FeedForwardLayer_1_3/bias/Read/ReadVariableOpReadVariableOp7transformer_encoder_5/Encoder-FeedForwardLayer_1_3/bias*
_output_shapes
:	*
dtype0
�
9transformer_encoder_5/Encoder-FeedForwardLayer_1_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:		*J
shared_name;9transformer_encoder_5/Encoder-FeedForwardLayer_1_3/kernel
�
Mtransformer_encoder_5/Encoder-FeedForwardLayer_1_3/kernel/Read/ReadVariableOpReadVariableOp9transformer_encoder_5/Encoder-FeedForwardLayer_1_3/kernel*
_output_shapes

:		*
dtype0
�
;transformer_encoder_5/Encoder-2nd-NormalizationLayer-3/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*L
shared_name=;transformer_encoder_5/Encoder-2nd-NormalizationLayer-3/beta
�
Otransformer_encoder_5/Encoder-2nd-NormalizationLayer-3/beta/Read/ReadVariableOpReadVariableOp;transformer_encoder_5/Encoder-2nd-NormalizationLayer-3/beta*
_output_shapes
:	*
dtype0
�
<transformer_encoder_5/Encoder-2nd-NormalizationLayer-3/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*M
shared_name><transformer_encoder_5/Encoder-2nd-NormalizationLayer-3/gamma
�
Ptransformer_encoder_5/Encoder-2nd-NormalizationLayer-3/gamma/Read/ReadVariableOpReadVariableOp<transformer_encoder_5/Encoder-2nd-NormalizationLayer-3/gamma*
_output_shapes
:	*
dtype0
�
;transformer_encoder_5/Encoder-1st-NormalizationLayer-3/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*L
shared_name=;transformer_encoder_5/Encoder-1st-NormalizationLayer-3/beta
�
Otransformer_encoder_5/Encoder-1st-NormalizationLayer-3/beta/Read/ReadVariableOpReadVariableOp;transformer_encoder_5/Encoder-1st-NormalizationLayer-3/beta*
_output_shapes
:	*
dtype0
�
<transformer_encoder_5/Encoder-1st-NormalizationLayer-3/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*M
shared_name><transformer_encoder_5/Encoder-1st-NormalizationLayer-3/gamma
�
Ptransformer_encoder_5/Encoder-1st-NormalizationLayer-3/gamma/Read/ReadVariableOpReadVariableOp<transformer_encoder_5/Encoder-1st-NormalizationLayer-3/gamma*
_output_shapes
:	*
dtype0
�
Htransformer_encoder_5/Encoder-SelfAttentionLayer-3/attention_output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*Y
shared_nameJHtransformer_encoder_5/Encoder-SelfAttentionLayer-3/attention_output/bias
�
\transformer_encoder_5/Encoder-SelfAttentionLayer-3/attention_output/bias/Read/ReadVariableOpReadVariableOpHtransformer_encoder_5/Encoder-SelfAttentionLayer-3/attention_output/bias*
_output_shapes
:	*
dtype0
�
Jtransformer_encoder_5/Encoder-SelfAttentionLayer-3/attention_output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*[
shared_nameLJtransformer_encoder_5/Encoder-SelfAttentionLayer-3/attention_output/kernel
�
^transformer_encoder_5/Encoder-SelfAttentionLayer-3/attention_output/kernel/Read/ReadVariableOpReadVariableOpJtransformer_encoder_5/Encoder-SelfAttentionLayer-3/attention_output/kernel*"
_output_shapes
:	*
dtype0
�
=transformer_encoder_5/Encoder-SelfAttentionLayer-3/value/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*N
shared_name?=transformer_encoder_5/Encoder-SelfAttentionLayer-3/value/bias
�
Qtransformer_encoder_5/Encoder-SelfAttentionLayer-3/value/bias/Read/ReadVariableOpReadVariableOp=transformer_encoder_5/Encoder-SelfAttentionLayer-3/value/bias*
_output_shapes

:*
dtype0
�
?transformer_encoder_5/Encoder-SelfAttentionLayer-3/value/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*P
shared_nameA?transformer_encoder_5/Encoder-SelfAttentionLayer-3/value/kernel
�
Stransformer_encoder_5/Encoder-SelfAttentionLayer-3/value/kernel/Read/ReadVariableOpReadVariableOp?transformer_encoder_5/Encoder-SelfAttentionLayer-3/value/kernel*"
_output_shapes
:	*
dtype0
�
;transformer_encoder_5/Encoder-SelfAttentionLayer-3/key/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*L
shared_name=;transformer_encoder_5/Encoder-SelfAttentionLayer-3/key/bias
�
Otransformer_encoder_5/Encoder-SelfAttentionLayer-3/key/bias/Read/ReadVariableOpReadVariableOp;transformer_encoder_5/Encoder-SelfAttentionLayer-3/key/bias*
_output_shapes

:*
dtype0
�
=transformer_encoder_5/Encoder-SelfAttentionLayer-3/key/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*N
shared_name?=transformer_encoder_5/Encoder-SelfAttentionLayer-3/key/kernel
�
Qtransformer_encoder_5/Encoder-SelfAttentionLayer-3/key/kernel/Read/ReadVariableOpReadVariableOp=transformer_encoder_5/Encoder-SelfAttentionLayer-3/key/kernel*"
_output_shapes
:	*
dtype0
�
=transformer_encoder_5/Encoder-SelfAttentionLayer-3/query/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*N
shared_name?=transformer_encoder_5/Encoder-SelfAttentionLayer-3/query/bias
�
Qtransformer_encoder_5/Encoder-SelfAttentionLayer-3/query/bias/Read/ReadVariableOpReadVariableOp=transformer_encoder_5/Encoder-SelfAttentionLayer-3/query/bias*
_output_shapes

:*
dtype0
�
?transformer_encoder_5/Encoder-SelfAttentionLayer-3/query/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*P
shared_nameA?transformer_encoder_5/Encoder-SelfAttentionLayer-3/query/kernel
�
Stransformer_encoder_5/Encoder-SelfAttentionLayer-3/query/kernel/Read/ReadVariableOpReadVariableOp?transformer_encoder_5/Encoder-SelfAttentionLayer-3/query/kernel*"
_output_shapes
:	*
dtype0
�
7transformer_encoder_4/Encoder-FeedForwardLayer_2_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*H
shared_name97transformer_encoder_4/Encoder-FeedForwardLayer_2_2/bias
�
Ktransformer_encoder_4/Encoder-FeedForwardLayer_2_2/bias/Read/ReadVariableOpReadVariableOp7transformer_encoder_4/Encoder-FeedForwardLayer_2_2/bias*
_output_shapes
:	*
dtype0
�
9transformer_encoder_4/Encoder-FeedForwardLayer_2_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:		*J
shared_name;9transformer_encoder_4/Encoder-FeedForwardLayer_2_2/kernel
�
Mtransformer_encoder_4/Encoder-FeedForwardLayer_2_2/kernel/Read/ReadVariableOpReadVariableOp9transformer_encoder_4/Encoder-FeedForwardLayer_2_2/kernel*
_output_shapes

:		*
dtype0
�
7transformer_encoder_4/Encoder-FeedForwardLayer_1_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*H
shared_name97transformer_encoder_4/Encoder-FeedForwardLayer_1_2/bias
�
Ktransformer_encoder_4/Encoder-FeedForwardLayer_1_2/bias/Read/ReadVariableOpReadVariableOp7transformer_encoder_4/Encoder-FeedForwardLayer_1_2/bias*
_output_shapes
:	*
dtype0
�
9transformer_encoder_4/Encoder-FeedForwardLayer_1_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:		*J
shared_name;9transformer_encoder_4/Encoder-FeedForwardLayer_1_2/kernel
�
Mtransformer_encoder_4/Encoder-FeedForwardLayer_1_2/kernel/Read/ReadVariableOpReadVariableOp9transformer_encoder_4/Encoder-FeedForwardLayer_1_2/kernel*
_output_shapes

:		*
dtype0
�
;transformer_encoder_4/Encoder-2nd-NormalizationLayer-2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*L
shared_name=;transformer_encoder_4/Encoder-2nd-NormalizationLayer-2/beta
�
Otransformer_encoder_4/Encoder-2nd-NormalizationLayer-2/beta/Read/ReadVariableOpReadVariableOp;transformer_encoder_4/Encoder-2nd-NormalizationLayer-2/beta*
_output_shapes
:	*
dtype0
�
<transformer_encoder_4/Encoder-2nd-NormalizationLayer-2/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*M
shared_name><transformer_encoder_4/Encoder-2nd-NormalizationLayer-2/gamma
�
Ptransformer_encoder_4/Encoder-2nd-NormalizationLayer-2/gamma/Read/ReadVariableOpReadVariableOp<transformer_encoder_4/Encoder-2nd-NormalizationLayer-2/gamma*
_output_shapes
:	*
dtype0
�
;transformer_encoder_4/Encoder-1st-NormalizationLayer-2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*L
shared_name=;transformer_encoder_4/Encoder-1st-NormalizationLayer-2/beta
�
Otransformer_encoder_4/Encoder-1st-NormalizationLayer-2/beta/Read/ReadVariableOpReadVariableOp;transformer_encoder_4/Encoder-1st-NormalizationLayer-2/beta*
_output_shapes
:	*
dtype0
�
<transformer_encoder_4/Encoder-1st-NormalizationLayer-2/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*M
shared_name><transformer_encoder_4/Encoder-1st-NormalizationLayer-2/gamma
�
Ptransformer_encoder_4/Encoder-1st-NormalizationLayer-2/gamma/Read/ReadVariableOpReadVariableOp<transformer_encoder_4/Encoder-1st-NormalizationLayer-2/gamma*
_output_shapes
:	*
dtype0
�
Htransformer_encoder_4/Encoder-SelfAttentionLayer-2/attention_output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*Y
shared_nameJHtransformer_encoder_4/Encoder-SelfAttentionLayer-2/attention_output/bias
�
\transformer_encoder_4/Encoder-SelfAttentionLayer-2/attention_output/bias/Read/ReadVariableOpReadVariableOpHtransformer_encoder_4/Encoder-SelfAttentionLayer-2/attention_output/bias*
_output_shapes
:	*
dtype0
�
Jtransformer_encoder_4/Encoder-SelfAttentionLayer-2/attention_output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*[
shared_nameLJtransformer_encoder_4/Encoder-SelfAttentionLayer-2/attention_output/kernel
�
^transformer_encoder_4/Encoder-SelfAttentionLayer-2/attention_output/kernel/Read/ReadVariableOpReadVariableOpJtransformer_encoder_4/Encoder-SelfAttentionLayer-2/attention_output/kernel*"
_output_shapes
:	*
dtype0
�
=transformer_encoder_4/Encoder-SelfAttentionLayer-2/value/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*N
shared_name?=transformer_encoder_4/Encoder-SelfAttentionLayer-2/value/bias
�
Qtransformer_encoder_4/Encoder-SelfAttentionLayer-2/value/bias/Read/ReadVariableOpReadVariableOp=transformer_encoder_4/Encoder-SelfAttentionLayer-2/value/bias*
_output_shapes

:*
dtype0
�
?transformer_encoder_4/Encoder-SelfAttentionLayer-2/value/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*P
shared_nameA?transformer_encoder_4/Encoder-SelfAttentionLayer-2/value/kernel
�
Stransformer_encoder_4/Encoder-SelfAttentionLayer-2/value/kernel/Read/ReadVariableOpReadVariableOp?transformer_encoder_4/Encoder-SelfAttentionLayer-2/value/kernel*"
_output_shapes
:	*
dtype0
�
;transformer_encoder_4/Encoder-SelfAttentionLayer-2/key/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*L
shared_name=;transformer_encoder_4/Encoder-SelfAttentionLayer-2/key/bias
�
Otransformer_encoder_4/Encoder-SelfAttentionLayer-2/key/bias/Read/ReadVariableOpReadVariableOp;transformer_encoder_4/Encoder-SelfAttentionLayer-2/key/bias*
_output_shapes

:*
dtype0
�
=transformer_encoder_4/Encoder-SelfAttentionLayer-2/key/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*N
shared_name?=transformer_encoder_4/Encoder-SelfAttentionLayer-2/key/kernel
�
Qtransformer_encoder_4/Encoder-SelfAttentionLayer-2/key/kernel/Read/ReadVariableOpReadVariableOp=transformer_encoder_4/Encoder-SelfAttentionLayer-2/key/kernel*"
_output_shapes
:	*
dtype0
�
=transformer_encoder_4/Encoder-SelfAttentionLayer-2/query/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*N
shared_name?=transformer_encoder_4/Encoder-SelfAttentionLayer-2/query/bias
�
Qtransformer_encoder_4/Encoder-SelfAttentionLayer-2/query/bias/Read/ReadVariableOpReadVariableOp=transformer_encoder_4/Encoder-SelfAttentionLayer-2/query/bias*
_output_shapes

:*
dtype0
�
?transformer_encoder_4/Encoder-SelfAttentionLayer-2/query/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*P
shared_nameA?transformer_encoder_4/Encoder-SelfAttentionLayer-2/query/kernel
�
Stransformer_encoder_4/Encoder-SelfAttentionLayer-2/query/kernel/Read/ReadVariableOpReadVariableOp?transformer_encoder_4/Encoder-SelfAttentionLayer-2/query/kernel*"
_output_shapes
:	*
dtype0
�
7transformer_encoder_3/Encoder-FeedForwardLayer_2_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*H
shared_name97transformer_encoder_3/Encoder-FeedForwardLayer_2_1/bias
�
Ktransformer_encoder_3/Encoder-FeedForwardLayer_2_1/bias/Read/ReadVariableOpReadVariableOp7transformer_encoder_3/Encoder-FeedForwardLayer_2_1/bias*
_output_shapes
:	*
dtype0
�
9transformer_encoder_3/Encoder-FeedForwardLayer_2_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:		*J
shared_name;9transformer_encoder_3/Encoder-FeedForwardLayer_2_1/kernel
�
Mtransformer_encoder_3/Encoder-FeedForwardLayer_2_1/kernel/Read/ReadVariableOpReadVariableOp9transformer_encoder_3/Encoder-FeedForwardLayer_2_1/kernel*
_output_shapes

:		*
dtype0
�
7transformer_encoder_3/Encoder-FeedForwardLayer_1_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*H
shared_name97transformer_encoder_3/Encoder-FeedForwardLayer_1_1/bias
�
Ktransformer_encoder_3/Encoder-FeedForwardLayer_1_1/bias/Read/ReadVariableOpReadVariableOp7transformer_encoder_3/Encoder-FeedForwardLayer_1_1/bias*
_output_shapes
:	*
dtype0
�
9transformer_encoder_3/Encoder-FeedForwardLayer_1_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:		*J
shared_name;9transformer_encoder_3/Encoder-FeedForwardLayer_1_1/kernel
�
Mtransformer_encoder_3/Encoder-FeedForwardLayer_1_1/kernel/Read/ReadVariableOpReadVariableOp9transformer_encoder_3/Encoder-FeedForwardLayer_1_1/kernel*
_output_shapes

:		*
dtype0
�
;transformer_encoder_3/Encoder-2nd-NormalizationLayer-1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*L
shared_name=;transformer_encoder_3/Encoder-2nd-NormalizationLayer-1/beta
�
Otransformer_encoder_3/Encoder-2nd-NormalizationLayer-1/beta/Read/ReadVariableOpReadVariableOp;transformer_encoder_3/Encoder-2nd-NormalizationLayer-1/beta*
_output_shapes
:	*
dtype0
�
<transformer_encoder_3/Encoder-2nd-NormalizationLayer-1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*M
shared_name><transformer_encoder_3/Encoder-2nd-NormalizationLayer-1/gamma
�
Ptransformer_encoder_3/Encoder-2nd-NormalizationLayer-1/gamma/Read/ReadVariableOpReadVariableOp<transformer_encoder_3/Encoder-2nd-NormalizationLayer-1/gamma*
_output_shapes
:	*
dtype0
�
;transformer_encoder_3/Encoder-1st-NormalizationLayer-1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*L
shared_name=;transformer_encoder_3/Encoder-1st-NormalizationLayer-1/beta
�
Otransformer_encoder_3/Encoder-1st-NormalizationLayer-1/beta/Read/ReadVariableOpReadVariableOp;transformer_encoder_3/Encoder-1st-NormalizationLayer-1/beta*
_output_shapes
:	*
dtype0
�
<transformer_encoder_3/Encoder-1st-NormalizationLayer-1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*M
shared_name><transformer_encoder_3/Encoder-1st-NormalizationLayer-1/gamma
�
Ptransformer_encoder_3/Encoder-1st-NormalizationLayer-1/gamma/Read/ReadVariableOpReadVariableOp<transformer_encoder_3/Encoder-1st-NormalizationLayer-1/gamma*
_output_shapes
:	*
dtype0
�
Htransformer_encoder_3/Encoder-SelfAttentionLayer-1/attention_output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*Y
shared_nameJHtransformer_encoder_3/Encoder-SelfAttentionLayer-1/attention_output/bias
�
\transformer_encoder_3/Encoder-SelfAttentionLayer-1/attention_output/bias/Read/ReadVariableOpReadVariableOpHtransformer_encoder_3/Encoder-SelfAttentionLayer-1/attention_output/bias*
_output_shapes
:	*
dtype0
�
Jtransformer_encoder_3/Encoder-SelfAttentionLayer-1/attention_output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*[
shared_nameLJtransformer_encoder_3/Encoder-SelfAttentionLayer-1/attention_output/kernel
�
^transformer_encoder_3/Encoder-SelfAttentionLayer-1/attention_output/kernel/Read/ReadVariableOpReadVariableOpJtransformer_encoder_3/Encoder-SelfAttentionLayer-1/attention_output/kernel*"
_output_shapes
:	*
dtype0
�
=transformer_encoder_3/Encoder-SelfAttentionLayer-1/value/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*N
shared_name?=transformer_encoder_3/Encoder-SelfAttentionLayer-1/value/bias
�
Qtransformer_encoder_3/Encoder-SelfAttentionLayer-1/value/bias/Read/ReadVariableOpReadVariableOp=transformer_encoder_3/Encoder-SelfAttentionLayer-1/value/bias*
_output_shapes

:*
dtype0
�
?transformer_encoder_3/Encoder-SelfAttentionLayer-1/value/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*P
shared_nameA?transformer_encoder_3/Encoder-SelfAttentionLayer-1/value/kernel
�
Stransformer_encoder_3/Encoder-SelfAttentionLayer-1/value/kernel/Read/ReadVariableOpReadVariableOp?transformer_encoder_3/Encoder-SelfAttentionLayer-1/value/kernel*"
_output_shapes
:	*
dtype0
�
;transformer_encoder_3/Encoder-SelfAttentionLayer-1/key/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*L
shared_name=;transformer_encoder_3/Encoder-SelfAttentionLayer-1/key/bias
�
Otransformer_encoder_3/Encoder-SelfAttentionLayer-1/key/bias/Read/ReadVariableOpReadVariableOp;transformer_encoder_3/Encoder-SelfAttentionLayer-1/key/bias*
_output_shapes

:*
dtype0
�
=transformer_encoder_3/Encoder-SelfAttentionLayer-1/key/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*N
shared_name?=transformer_encoder_3/Encoder-SelfAttentionLayer-1/key/kernel
�
Qtransformer_encoder_3/Encoder-SelfAttentionLayer-1/key/kernel/Read/ReadVariableOpReadVariableOp=transformer_encoder_3/Encoder-SelfAttentionLayer-1/key/kernel*"
_output_shapes
:	*
dtype0
�
=transformer_encoder_3/Encoder-SelfAttentionLayer-1/query/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*N
shared_name?=transformer_encoder_3/Encoder-SelfAttentionLayer-1/query/bias
�
Qtransformer_encoder_3/Encoder-SelfAttentionLayer-1/query/bias/Read/ReadVariableOpReadVariableOp=transformer_encoder_3/Encoder-SelfAttentionLayer-1/query/bias*
_output_shapes

:*
dtype0
�
?transformer_encoder_3/Encoder-SelfAttentionLayer-1/query/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*P
shared_nameA?transformer_encoder_3/Encoder-SelfAttentionLayer-1/query/kernel
�
Stransformer_encoder_3/Encoder-SelfAttentionLayer-1/query/kernel/Read/ReadVariableOpReadVariableOp?transformer_encoder_3/Encoder-SelfAttentionLayer-1/query/kernel*"
_output_shapes
:	*
dtype0
�
PredictionImprovement/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namePredictionImprovement/bias
�
.PredictionImprovement/bias/Read/ReadVariableOpReadVariableOpPredictionImprovement/bias*
_output_shapes
:*
dtype0
�
PredictionImprovement/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*-
shared_namePredictionImprovement/kernel
�
0PredictionImprovement/kernel/Read/ReadVariableOpReadVariableOpPredictionImprovement/kernel*
_output_shapes

:
*
dtype0
�
#FullyConnectedLayerImprovement/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*4
shared_name%#FullyConnectedLayerImprovement/bias
�
7FullyConnectedLayerImprovement/bias/Read/ReadVariableOpReadVariableOp#FullyConnectedLayerImprovement/bias*
_output_shapes
:
*
dtype0
�
%FullyConnectedLayerImprovement/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:

*6
shared_name'%FullyConnectedLayerImprovement/kernel
�
9FullyConnectedLayerImprovement/kernel/Read/ReadVariableOpReadVariableOp%FullyConnectedLayerImprovement/kernel*
_output_shapes

:

*
dtype0
~
FinalLayerNorm/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*$
shared_nameFinalLayerNorm/beta
w
'FinalLayerNorm/beta/Read/ReadVariableOpReadVariableOpFinalLayerNorm/beta*
_output_shapes
:	*
dtype0
�
FinalLayerNorm/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*%
shared_nameFinalLayerNorm/gamma
y
(FinalLayerNorm/gamma/Read/ReadVariableOpReadVariableOpFinalLayerNorm/gamma*
_output_shapes
:	*
dtype0
�
'serving_default_StackLevelInputFeaturesPlaceholder*+
_output_shapes
:���������P	*
dtype0* 
shape:���������P	
�
serving_default_TimeLimitInputPlaceholder*'
_output_shapes
:���������*
dtype0*
shape:���������
�
StatefulPartitionedCallStatefulPartitionedCall'serving_default_StackLevelInputFeaturesserving_default_TimeLimitInput<transformer_encoder_3/Encoder-1st-NormalizationLayer-1/gamma;transformer_encoder_3/Encoder-1st-NormalizationLayer-1/beta?transformer_encoder_3/Encoder-SelfAttentionLayer-1/query/kernel=transformer_encoder_3/Encoder-SelfAttentionLayer-1/query/bias=transformer_encoder_3/Encoder-SelfAttentionLayer-1/key/kernel;transformer_encoder_3/Encoder-SelfAttentionLayer-1/key/bias?transformer_encoder_3/Encoder-SelfAttentionLayer-1/value/kernel=transformer_encoder_3/Encoder-SelfAttentionLayer-1/value/biasJtransformer_encoder_3/Encoder-SelfAttentionLayer-1/attention_output/kernelHtransformer_encoder_3/Encoder-SelfAttentionLayer-1/attention_output/bias<transformer_encoder_3/Encoder-2nd-NormalizationLayer-1/gamma;transformer_encoder_3/Encoder-2nd-NormalizationLayer-1/beta9transformer_encoder_3/Encoder-FeedForwardLayer_1_1/kernel7transformer_encoder_3/Encoder-FeedForwardLayer_1_1/bias9transformer_encoder_3/Encoder-FeedForwardLayer_2_1/kernel7transformer_encoder_3/Encoder-FeedForwardLayer_2_1/bias<transformer_encoder_4/Encoder-1st-NormalizationLayer-2/gamma;transformer_encoder_4/Encoder-1st-NormalizationLayer-2/beta?transformer_encoder_4/Encoder-SelfAttentionLayer-2/query/kernel=transformer_encoder_4/Encoder-SelfAttentionLayer-2/query/bias=transformer_encoder_4/Encoder-SelfAttentionLayer-2/key/kernel;transformer_encoder_4/Encoder-SelfAttentionLayer-2/key/bias?transformer_encoder_4/Encoder-SelfAttentionLayer-2/value/kernel=transformer_encoder_4/Encoder-SelfAttentionLayer-2/value/biasJtransformer_encoder_4/Encoder-SelfAttentionLayer-2/attention_output/kernelHtransformer_encoder_4/Encoder-SelfAttentionLayer-2/attention_output/bias<transformer_encoder_4/Encoder-2nd-NormalizationLayer-2/gamma;transformer_encoder_4/Encoder-2nd-NormalizationLayer-2/beta9transformer_encoder_4/Encoder-FeedForwardLayer_1_2/kernel7transformer_encoder_4/Encoder-FeedForwardLayer_1_2/bias9transformer_encoder_4/Encoder-FeedForwardLayer_2_2/kernel7transformer_encoder_4/Encoder-FeedForwardLayer_2_2/bias<transformer_encoder_5/Encoder-1st-NormalizationLayer-3/gamma;transformer_encoder_5/Encoder-1st-NormalizationLayer-3/beta?transformer_encoder_5/Encoder-SelfAttentionLayer-3/query/kernel=transformer_encoder_5/Encoder-SelfAttentionLayer-3/query/bias=transformer_encoder_5/Encoder-SelfAttentionLayer-3/key/kernel;transformer_encoder_5/Encoder-SelfAttentionLayer-3/key/bias?transformer_encoder_5/Encoder-SelfAttentionLayer-3/value/kernel=transformer_encoder_5/Encoder-SelfAttentionLayer-3/value/biasJtransformer_encoder_5/Encoder-SelfAttentionLayer-3/attention_output/kernelHtransformer_encoder_5/Encoder-SelfAttentionLayer-3/attention_output/bias<transformer_encoder_5/Encoder-2nd-NormalizationLayer-3/gamma;transformer_encoder_5/Encoder-2nd-NormalizationLayer-3/beta9transformer_encoder_5/Encoder-FeedForwardLayer_1_3/kernel7transformer_encoder_5/Encoder-FeedForwardLayer_1_3/bias9transformer_encoder_5/Encoder-FeedForwardLayer_2_3/kernel7transformer_encoder_5/Encoder-FeedForwardLayer_2_3/biasFinalLayerNorm/gammaFinalLayerNorm/beta%FullyConnectedLayerImprovement/kernel#FullyConnectedLayerImprovement/biasPredictionImprovement/kernelPredictionImprovement/bias*C
Tin<
:28*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:*X
_read_only_resource_inputs:
86	
 !"#$%&'()*+,-./01234567*0
config_proto 

CPU

GPU2*0J 8� *-
f(R&
$__inference_signature_wrapper_231873

NoOpNoOp
��
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*��
value��B�� B��
�
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer-6
layer-7
	layer-8

layer-9
layer_with_weights-4
layer-10
layer_with_weights-5
layer-11
layer-12
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures*
* 
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses* 
�
	variables
trainable_variables
regularization_losses
	keras_api
 __call__
*!&call_and_return_all_conditional_losses
"self_attention_layer
#add1
$add2
%
layernorm1
&
layernorm2
'feed_forward_layer_1
(feed_forward_layer_2
)dropout_layer*
�
*	variables
+trainable_variables
,regularization_losses
-	keras_api
.__call__
*/&call_and_return_all_conditional_losses
0self_attention_layer
1add1
2add2
3
layernorm1
4
layernorm2
5feed_forward_layer_1
6feed_forward_layer_2
7dropout_layer*
�
8	variables
9trainable_variables
:regularization_losses
;	keras_api
<__call__
*=&call_and_return_all_conditional_losses
>self_attention_layer
?add1
@add2
A
layernorm1
B
layernorm2
Cfeed_forward_layer_1
Dfeed_forward_layer_2
Edropout_layer*
�
F	variables
Gtrainable_variables
Hregularization_losses
I	keras_api
J__call__
*K&call_and_return_all_conditional_losses
Laxis
	Mgamma
Nbeta*
* 
�
O	variables
Ptrainable_variables
Qregularization_losses
R	keras_api
S__call__
*T&call_and_return_all_conditional_losses* 
�
U	variables
Vtrainable_variables
Wregularization_losses
X	keras_api
Y__call__
*Z&call_and_return_all_conditional_losses* 
�
[	variables
\trainable_variables
]regularization_losses
^	keras_api
___call__
*`&call_and_return_all_conditional_losses* 
�
a	variables
btrainable_variables
cregularization_losses
d	keras_api
e__call__
*f&call_and_return_all_conditional_losses

gkernel
hbias*
�
i	variables
jtrainable_variables
kregularization_losses
l	keras_api
m__call__
*n&call_and_return_all_conditional_losses

okernel
pbias*
�
q	variables
rtrainable_variables
sregularization_losses
t	keras_api
u__call__
*v&call_and_return_all_conditional_losses* 
�
w0
x1
y2
z3
{4
|5
}6
~7
8
�9
�10
�11
�12
�13
�14
�15
�16
�17
�18
�19
�20
�21
�22
�23
�24
�25
�26
�27
�28
�29
�30
�31
�32
�33
�34
�35
�36
�37
�38
�39
�40
�41
�42
�43
�44
�45
�46
�47
M48
N49
g50
h51
o52
p53*
�
w0
x1
y2
z3
{4
|5
}6
~7
8
�9
�10
�11
�12
�13
�14
�15
�16
�17
�18
�19
�20
�21
�22
�23
�24
�25
�26
�27
�28
�29
�30
�31
�32
�33
�34
�35
�36
�37
�38
�39
�40
�41
�42
�43
�44
�45
�46
�47
M48
N49
g50
h51
o52
p53*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 

�serving_default* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 
�
w0
x1
y2
z3
{4
|5
}6
~7
8
�9
�10
�11
�12
�13
�14
�15*
�
w0
x1
y2
z3
{4
|5
}6
~7
8
�9
�10
�11
�12
�13
�14
�15*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
 __call__
*!&call_and_return_all_conditional_losses
&!"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�_query_dense
�
_key_dense
�_value_dense
�_softmax
�_dropout_layer
�_output_dense*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis
	gamma
	�beta*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis

�gamma
	�beta*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�_random_generator* 
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15*
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
*	variables
+trainable_variables
,regularization_losses
.__call__
*/&call_and_return_all_conditional_losses
&/"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�_query_dense
�
_key_dense
�_value_dense
�_softmax
�_dropout_layer
�_output_dense*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis

�gamma
	�beta*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis

�gamma
	�beta*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�_random_generator* 
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15*
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
8	variables
9trainable_variables
:regularization_losses
<__call__
*=&call_and_return_all_conditional_losses
&="call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�_query_dense
�
_key_dense
�_value_dense
�_softmax
�_dropout_layer
�_output_dense*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis

�gamma
	�beta*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis

�gamma
	�beta*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�_random_generator* 

M0
N1*

M0
N1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
F	variables
Gtrainable_variables
Hregularization_losses
J__call__
*K&call_and_return_all_conditional_losses
&K"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 
c]
VARIABLE_VALUEFinalLayerNorm/gamma5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEFinalLayerNorm/beta4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
O	variables
Ptrainable_variables
Qregularization_losses
S__call__
*T&call_and_return_all_conditional_losses
&T"call_and_return_conditional_losses* 

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
U	variables
Vtrainable_variables
Wregularization_losses
Y__call__
*Z&call_and_return_all_conditional_losses
&Z"call_and_return_conditional_losses* 

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
[	variables
\trainable_variables
]regularization_losses
___call__
*`&call_and_return_all_conditional_losses
&`"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

g0
h1*

g0
h1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
a	variables
btrainable_variables
cregularization_losses
e__call__
*f&call_and_return_all_conditional_losses
&f"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
uo
VARIABLE_VALUE%FullyConnectedLayerImprovement/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUE#FullyConnectedLayerImprovement/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*

o0
p1*

o0
p1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
i	variables
jtrainable_variables
kregularization_losses
m__call__
*n&call_and_return_all_conditional_losses
&n"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
lf
VARIABLE_VALUEPredictionImprovement/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUEPredictionImprovement/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
q	variables
rtrainable_variables
sregularization_losses
u__call__
*v&call_and_return_all_conditional_losses
&v"call_and_return_conditional_losses* 

�trace_0
�trace_1* 

�trace_0
�trace_1* 
y
VARIABLE_VALUE?transformer_encoder_3/Encoder-SelfAttentionLayer-1/query/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUE=transformer_encoder_3/Encoder-SelfAttentionLayer-1/query/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUE=transformer_encoder_3/Encoder-SelfAttentionLayer-1/key/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUE;transformer_encoder_3/Encoder-SelfAttentionLayer-1/key/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE?transformer_encoder_3/Encoder-SelfAttentionLayer-1/value/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUE=transformer_encoder_3/Encoder-SelfAttentionLayer-1/value/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEJtransformer_encoder_3/Encoder-SelfAttentionLayer-1/attention_output/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEHtransformer_encoder_3/Encoder-SelfAttentionLayer-1/attention_output/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUE<transformer_encoder_3/Encoder-1st-NormalizationLayer-1/gamma&variables/8/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUE;transformer_encoder_3/Encoder-1st-NormalizationLayer-1/beta&variables/9/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUE<transformer_encoder_3/Encoder-2nd-NormalizationLayer-1/gamma'variables/10/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUE;transformer_encoder_3/Encoder-2nd-NormalizationLayer-1/beta'variables/11/.ATTRIBUTES/VARIABLE_VALUE*
zt
VARIABLE_VALUE9transformer_encoder_3/Encoder-FeedForwardLayer_1_1/kernel'variables/12/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUE7transformer_encoder_3/Encoder-FeedForwardLayer_1_1/bias'variables/13/.ATTRIBUTES/VARIABLE_VALUE*
zt
VARIABLE_VALUE9transformer_encoder_3/Encoder-FeedForwardLayer_2_1/kernel'variables/14/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUE7transformer_encoder_3/Encoder-FeedForwardLayer_2_1/bias'variables/15/.ATTRIBUTES/VARIABLE_VALUE*
�z
VARIABLE_VALUE?transformer_encoder_4/Encoder-SelfAttentionLayer-2/query/kernel'variables/16/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE=transformer_encoder_4/Encoder-SelfAttentionLayer-2/query/bias'variables/17/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE=transformer_encoder_4/Encoder-SelfAttentionLayer-2/key/kernel'variables/18/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUE;transformer_encoder_4/Encoder-SelfAttentionLayer-2/key/bias'variables/19/.ATTRIBUTES/VARIABLE_VALUE*
�z
VARIABLE_VALUE?transformer_encoder_4/Encoder-SelfAttentionLayer-2/value/kernel'variables/20/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE=transformer_encoder_4/Encoder-SelfAttentionLayer-2/value/bias'variables/21/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEJtransformer_encoder_4/Encoder-SelfAttentionLayer-2/attention_output/kernel'variables/22/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEHtransformer_encoder_4/Encoder-SelfAttentionLayer-2/attention_output/bias'variables/23/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUE<transformer_encoder_4/Encoder-1st-NormalizationLayer-2/gamma'variables/24/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUE;transformer_encoder_4/Encoder-1st-NormalizationLayer-2/beta'variables/25/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUE<transformer_encoder_4/Encoder-2nd-NormalizationLayer-2/gamma'variables/26/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUE;transformer_encoder_4/Encoder-2nd-NormalizationLayer-2/beta'variables/27/.ATTRIBUTES/VARIABLE_VALUE*
zt
VARIABLE_VALUE9transformer_encoder_4/Encoder-FeedForwardLayer_1_2/kernel'variables/28/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUE7transformer_encoder_4/Encoder-FeedForwardLayer_1_2/bias'variables/29/.ATTRIBUTES/VARIABLE_VALUE*
zt
VARIABLE_VALUE9transformer_encoder_4/Encoder-FeedForwardLayer_2_2/kernel'variables/30/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUE7transformer_encoder_4/Encoder-FeedForwardLayer_2_2/bias'variables/31/.ATTRIBUTES/VARIABLE_VALUE*
�z
VARIABLE_VALUE?transformer_encoder_5/Encoder-SelfAttentionLayer-3/query/kernel'variables/32/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE=transformer_encoder_5/Encoder-SelfAttentionLayer-3/query/bias'variables/33/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE=transformer_encoder_5/Encoder-SelfAttentionLayer-3/key/kernel'variables/34/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUE;transformer_encoder_5/Encoder-SelfAttentionLayer-3/key/bias'variables/35/.ATTRIBUTES/VARIABLE_VALUE*
�z
VARIABLE_VALUE?transformer_encoder_5/Encoder-SelfAttentionLayer-3/value/kernel'variables/36/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE=transformer_encoder_5/Encoder-SelfAttentionLayer-3/value/bias'variables/37/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEJtransformer_encoder_5/Encoder-SelfAttentionLayer-3/attention_output/kernel'variables/38/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEHtransformer_encoder_5/Encoder-SelfAttentionLayer-3/attention_output/bias'variables/39/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUE<transformer_encoder_5/Encoder-1st-NormalizationLayer-3/gamma'variables/40/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUE;transformer_encoder_5/Encoder-1st-NormalizationLayer-3/beta'variables/41/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUE<transformer_encoder_5/Encoder-2nd-NormalizationLayer-3/gamma'variables/42/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUE;transformer_encoder_5/Encoder-2nd-NormalizationLayer-3/beta'variables/43/.ATTRIBUTES/VARIABLE_VALUE*
zt
VARIABLE_VALUE9transformer_encoder_5/Encoder-FeedForwardLayer_1_3/kernel'variables/44/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUE7transformer_encoder_5/Encoder-FeedForwardLayer_1_3/bias'variables/45/.ATTRIBUTES/VARIABLE_VALUE*
zt
VARIABLE_VALUE9transformer_encoder_5/Encoder-FeedForwardLayer_2_3/kernel'variables/46/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUE7transformer_encoder_5/Encoder-FeedForwardLayer_2_3/bias'variables/47/.ATTRIBUTES/VARIABLE_VALUE*
* 
b
0
1
2
3
4
5
6
7
	8

9
10
11
12*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
<
"0
#1
$2
%3
&4
'5
(6
)7*
* 
* 
* 
* 
* 
* 
* 
<
w0
x1
y2
z3
{4
|5
}6
~7*
<
w0
x1
y2
z3
{4
|5
}6
~7*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�partial_output_shape
�full_output_shape

wkernel
xbias*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�partial_output_shape
�full_output_shape

ykernel
zbias*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�partial_output_shape
�full_output_shape

{kernel
|bias*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�_random_generator* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�partial_output_shape
�full_output_shape

}kernel
~bias*
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 
* 
* 

0
�1*

0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 
* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 
* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 
* 
* 
* 
* 
<
00
11
22
33
44
55
66
77*
* 
* 
* 
* 
* 
* 
* 
D
�0
�1
�2
�3
�4
�5
�6
�7*
D
�0
�1
�2
�3
�4
�5
�6
�7*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�partial_output_shape
�full_output_shape
�kernel
	�bias*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�partial_output_shape
�full_output_shape
�kernel
	�bias*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�partial_output_shape
�full_output_shape
�kernel
	�bias*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�_random_generator* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�partial_output_shape
�full_output_shape
�kernel
	�bias*
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 
* 
* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 
* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 
* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 
* 
* 
* 
* 
<
>0
?1
@2
A3
B4
C5
D6
E7*
* 
* 
* 
* 
* 
* 
* 
D
�0
�1
�2
�3
�4
�5
�6
�7*
D
�0
�1
�2
�3
�4
�5
�6
�7*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�partial_output_shape
�full_output_shape
�kernel
	�bias*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�partial_output_shape
�full_output_shape
�kernel
	�bias*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�partial_output_shape
�full_output_shape
�kernel
	�bias*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�_random_generator* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�partial_output_shape
�full_output_shape
�kernel
	�bias*
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 
* 
* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 
* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 
* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
4
�0
�1
�2
�3
�4
�5*
* 
* 
* 

w0
x1*

w0
x1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 
* 
* 

y0
z1*

y0
z1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 
* 
* 

{0
|1*

{0
|1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 
* 
* 
* 

}0
~1*

}0
~1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
4
�0
�1
�2
�3
�4
�5*
* 
* 
* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 
* 
* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 
* 
* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 
* 
* 
* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
4
�0
�1
�2
�3
�4
�5*
* 
* 
* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 
* 
* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 
* 
* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 
* 
* 
* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameFinalLayerNorm/gammaFinalLayerNorm/beta%FullyConnectedLayerImprovement/kernel#FullyConnectedLayerImprovement/biasPredictionImprovement/kernelPredictionImprovement/bias?transformer_encoder_3/Encoder-SelfAttentionLayer-1/query/kernel=transformer_encoder_3/Encoder-SelfAttentionLayer-1/query/bias=transformer_encoder_3/Encoder-SelfAttentionLayer-1/key/kernel;transformer_encoder_3/Encoder-SelfAttentionLayer-1/key/bias?transformer_encoder_3/Encoder-SelfAttentionLayer-1/value/kernel=transformer_encoder_3/Encoder-SelfAttentionLayer-1/value/biasJtransformer_encoder_3/Encoder-SelfAttentionLayer-1/attention_output/kernelHtransformer_encoder_3/Encoder-SelfAttentionLayer-1/attention_output/bias<transformer_encoder_3/Encoder-1st-NormalizationLayer-1/gamma;transformer_encoder_3/Encoder-1st-NormalizationLayer-1/beta<transformer_encoder_3/Encoder-2nd-NormalizationLayer-1/gamma;transformer_encoder_3/Encoder-2nd-NormalizationLayer-1/beta9transformer_encoder_3/Encoder-FeedForwardLayer_1_1/kernel7transformer_encoder_3/Encoder-FeedForwardLayer_1_1/bias9transformer_encoder_3/Encoder-FeedForwardLayer_2_1/kernel7transformer_encoder_3/Encoder-FeedForwardLayer_2_1/bias?transformer_encoder_4/Encoder-SelfAttentionLayer-2/query/kernel=transformer_encoder_4/Encoder-SelfAttentionLayer-2/query/bias=transformer_encoder_4/Encoder-SelfAttentionLayer-2/key/kernel;transformer_encoder_4/Encoder-SelfAttentionLayer-2/key/bias?transformer_encoder_4/Encoder-SelfAttentionLayer-2/value/kernel=transformer_encoder_4/Encoder-SelfAttentionLayer-2/value/biasJtransformer_encoder_4/Encoder-SelfAttentionLayer-2/attention_output/kernelHtransformer_encoder_4/Encoder-SelfAttentionLayer-2/attention_output/bias<transformer_encoder_4/Encoder-1st-NormalizationLayer-2/gamma;transformer_encoder_4/Encoder-1st-NormalizationLayer-2/beta<transformer_encoder_4/Encoder-2nd-NormalizationLayer-2/gamma;transformer_encoder_4/Encoder-2nd-NormalizationLayer-2/beta9transformer_encoder_4/Encoder-FeedForwardLayer_1_2/kernel7transformer_encoder_4/Encoder-FeedForwardLayer_1_2/bias9transformer_encoder_4/Encoder-FeedForwardLayer_2_2/kernel7transformer_encoder_4/Encoder-FeedForwardLayer_2_2/bias?transformer_encoder_5/Encoder-SelfAttentionLayer-3/query/kernel=transformer_encoder_5/Encoder-SelfAttentionLayer-3/query/bias=transformer_encoder_5/Encoder-SelfAttentionLayer-3/key/kernel;transformer_encoder_5/Encoder-SelfAttentionLayer-3/key/bias?transformer_encoder_5/Encoder-SelfAttentionLayer-3/value/kernel=transformer_encoder_5/Encoder-SelfAttentionLayer-3/value/biasJtransformer_encoder_5/Encoder-SelfAttentionLayer-3/attention_output/kernelHtransformer_encoder_5/Encoder-SelfAttentionLayer-3/attention_output/bias<transformer_encoder_5/Encoder-1st-NormalizationLayer-3/gamma;transformer_encoder_5/Encoder-1st-NormalizationLayer-3/beta<transformer_encoder_5/Encoder-2nd-NormalizationLayer-3/gamma;transformer_encoder_5/Encoder-2nd-NormalizationLayer-3/beta9transformer_encoder_5/Encoder-FeedForwardLayer_1_3/kernel7transformer_encoder_5/Encoder-FeedForwardLayer_1_3/bias9transformer_encoder_5/Encoder-FeedForwardLayer_2_3/kernel7transformer_encoder_5/Encoder-FeedForwardLayer_2_3/biasConst*C
Tin<
:28*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *(
f#R!
__inference__traced_save_233739
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameFinalLayerNorm/gammaFinalLayerNorm/beta%FullyConnectedLayerImprovement/kernel#FullyConnectedLayerImprovement/biasPredictionImprovement/kernelPredictionImprovement/bias?transformer_encoder_3/Encoder-SelfAttentionLayer-1/query/kernel=transformer_encoder_3/Encoder-SelfAttentionLayer-1/query/bias=transformer_encoder_3/Encoder-SelfAttentionLayer-1/key/kernel;transformer_encoder_3/Encoder-SelfAttentionLayer-1/key/bias?transformer_encoder_3/Encoder-SelfAttentionLayer-1/value/kernel=transformer_encoder_3/Encoder-SelfAttentionLayer-1/value/biasJtransformer_encoder_3/Encoder-SelfAttentionLayer-1/attention_output/kernelHtransformer_encoder_3/Encoder-SelfAttentionLayer-1/attention_output/bias<transformer_encoder_3/Encoder-1st-NormalizationLayer-1/gamma;transformer_encoder_3/Encoder-1st-NormalizationLayer-1/beta<transformer_encoder_3/Encoder-2nd-NormalizationLayer-1/gamma;transformer_encoder_3/Encoder-2nd-NormalizationLayer-1/beta9transformer_encoder_3/Encoder-FeedForwardLayer_1_1/kernel7transformer_encoder_3/Encoder-FeedForwardLayer_1_1/bias9transformer_encoder_3/Encoder-FeedForwardLayer_2_1/kernel7transformer_encoder_3/Encoder-FeedForwardLayer_2_1/bias?transformer_encoder_4/Encoder-SelfAttentionLayer-2/query/kernel=transformer_encoder_4/Encoder-SelfAttentionLayer-2/query/bias=transformer_encoder_4/Encoder-SelfAttentionLayer-2/key/kernel;transformer_encoder_4/Encoder-SelfAttentionLayer-2/key/bias?transformer_encoder_4/Encoder-SelfAttentionLayer-2/value/kernel=transformer_encoder_4/Encoder-SelfAttentionLayer-2/value/biasJtransformer_encoder_4/Encoder-SelfAttentionLayer-2/attention_output/kernelHtransformer_encoder_4/Encoder-SelfAttentionLayer-2/attention_output/bias<transformer_encoder_4/Encoder-1st-NormalizationLayer-2/gamma;transformer_encoder_4/Encoder-1st-NormalizationLayer-2/beta<transformer_encoder_4/Encoder-2nd-NormalizationLayer-2/gamma;transformer_encoder_4/Encoder-2nd-NormalizationLayer-2/beta9transformer_encoder_4/Encoder-FeedForwardLayer_1_2/kernel7transformer_encoder_4/Encoder-FeedForwardLayer_1_2/bias9transformer_encoder_4/Encoder-FeedForwardLayer_2_2/kernel7transformer_encoder_4/Encoder-FeedForwardLayer_2_2/bias?transformer_encoder_5/Encoder-SelfAttentionLayer-3/query/kernel=transformer_encoder_5/Encoder-SelfAttentionLayer-3/query/bias=transformer_encoder_5/Encoder-SelfAttentionLayer-3/key/kernel;transformer_encoder_5/Encoder-SelfAttentionLayer-3/key/bias?transformer_encoder_5/Encoder-SelfAttentionLayer-3/value/kernel=transformer_encoder_5/Encoder-SelfAttentionLayer-3/value/biasJtransformer_encoder_5/Encoder-SelfAttentionLayer-3/attention_output/kernelHtransformer_encoder_5/Encoder-SelfAttentionLayer-3/attention_output/bias<transformer_encoder_5/Encoder-1st-NormalizationLayer-3/gamma;transformer_encoder_5/Encoder-1st-NormalizationLayer-3/beta<transformer_encoder_5/Encoder-2nd-NormalizationLayer-3/gamma;transformer_encoder_5/Encoder-2nd-NormalizationLayer-3/beta9transformer_encoder_5/Encoder-FeedForwardLayer_1_3/kernel7transformer_encoder_5/Encoder-FeedForwardLayer_1_3/bias9transformer_encoder_5/Encoder-FeedForwardLayer_2_3/kernel7transformer_encoder_5/Encoder-FeedForwardLayer_2_3/bias*B
Tin;
927*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *+
f&R$
"__inference__traced_restore_233910�.
�e
�
C__inference_model_1_layer_call_and_return_conditional_losses_231228
stacklevelinputfeatures
timelimitinput*
transformer_encoder_3_230736:	*
transformer_encoder_3_230738:	2
transformer_encoder_3_230740:	.
transformer_encoder_3_230742:2
transformer_encoder_3_230744:	.
transformer_encoder_3_230746:2
transformer_encoder_3_230748:	.
transformer_encoder_3_230750:2
transformer_encoder_3_230752:	*
transformer_encoder_3_230754:	*
transformer_encoder_3_230756:	*
transformer_encoder_3_230758:	.
transformer_encoder_3_230760:		*
transformer_encoder_3_230762:	.
transformer_encoder_3_230764:		*
transformer_encoder_3_230766:	*
transformer_encoder_4_230944:	*
transformer_encoder_4_230946:	2
transformer_encoder_4_230948:	.
transformer_encoder_4_230950:2
transformer_encoder_4_230952:	.
transformer_encoder_4_230954:2
transformer_encoder_4_230956:	.
transformer_encoder_4_230958:2
transformer_encoder_4_230960:	*
transformer_encoder_4_230962:	*
transformer_encoder_4_230964:	*
transformer_encoder_4_230966:	.
transformer_encoder_4_230968:		*
transformer_encoder_4_230970:	.
transformer_encoder_4_230972:		*
transformer_encoder_4_230974:	*
transformer_encoder_5_231152:	*
transformer_encoder_5_231154:	2
transformer_encoder_5_231156:	.
transformer_encoder_5_231158:2
transformer_encoder_5_231160:	.
transformer_encoder_5_231162:2
transformer_encoder_5_231164:	.
transformer_encoder_5_231166:2
transformer_encoder_5_231168:	*
transformer_encoder_5_231170:	*
transformer_encoder_5_231172:	*
transformer_encoder_5_231174:	.
transformer_encoder_5_231176:		*
transformer_encoder_5_231178:	.
transformer_encoder_5_231180:		*
transformer_encoder_5_231182:	#
finallayernorm_231186:	#
finallayernorm_231188:	7
%fullyconnectedlayerimprovement_231211:

3
%fullyconnectedlayerimprovement_231213:
.
predictionimprovement_231216:
*
predictionimprovement_231218:
identity��&FinalLayerNorm/StatefulPartitionedCall�6FullyConnectedLayerImprovement/StatefulPartitionedCall�-PredictionImprovement/StatefulPartitionedCall�-transformer_encoder_3/StatefulPartitionedCall�-transformer_encoder_4/StatefulPartitionedCall�-transformer_encoder_5/StatefulPartitionedCall�
MaskingLayer/PartitionedCallPartitionedCallstacklevelinputfeatures*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������P	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_MaskingLayer_layer_call_and_return_conditional_losses_229767�
transformer_encoder_3/CastCast%MaskingLayer/PartitionedCall:output:0*

DstT0*

SrcT0*+
_output_shapes
:���������P	�
-transformer_encoder_3/StatefulPartitionedCallStatefulPartitionedCalltransformer_encoder_3/Cast:y:0transformer_encoder_3_230736transformer_encoder_3_230738transformer_encoder_3_230740transformer_encoder_3_230742transformer_encoder_3_230744transformer_encoder_3_230746transformer_encoder_3_230748transformer_encoder_3_230750transformer_encoder_3_230752transformer_encoder_3_230754transformer_encoder_3_230756transformer_encoder_3_230758transformer_encoder_3_230760transformer_encoder_3_230762transformer_encoder_3_230764transformer_encoder_3_230766*
Tin
2*
Tout
2*
_collective_manager_ids
 *F
_output_shapes4
2:���������P	:���������PP*2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *Z
fURS
Q__inference_transformer_encoder_3_layer_call_and_return_conditional_losses_230735�
-transformer_encoder_4/StatefulPartitionedCallStatefulPartitionedCall6transformer_encoder_3/StatefulPartitionedCall:output:0transformer_encoder_4_230944transformer_encoder_4_230946transformer_encoder_4_230948transformer_encoder_4_230950transformer_encoder_4_230952transformer_encoder_4_230954transformer_encoder_4_230956transformer_encoder_4_230958transformer_encoder_4_230960transformer_encoder_4_230962transformer_encoder_4_230964transformer_encoder_4_230966transformer_encoder_4_230968transformer_encoder_4_230970transformer_encoder_4_230972transformer_encoder_4_230974*
Tin
2*
Tout
2*
_collective_manager_ids
 *F
_output_shapes4
2:���������P	:���������PP*2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *Z
fURS
Q__inference_transformer_encoder_4_layer_call_and_return_conditional_losses_230943�
-transformer_encoder_5/StatefulPartitionedCallStatefulPartitionedCall6transformer_encoder_4/StatefulPartitionedCall:output:0transformer_encoder_5_231152transformer_encoder_5_231154transformer_encoder_5_231156transformer_encoder_5_231158transformer_encoder_5_231160transformer_encoder_5_231162transformer_encoder_5_231164transformer_encoder_5_231166transformer_encoder_5_231168transformer_encoder_5_231170transformer_encoder_5_231172transformer_encoder_5_231174transformer_encoder_5_231176transformer_encoder_5_231178transformer_encoder_5_231180transformer_encoder_5_231182*
Tin
2*
Tout
2*
_collective_manager_ids
 *F
_output_shapes4
2:���������P	:���������PP*2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *Z
fURS
Q__inference_transformer_encoder_5_layer_call_and_return_conditional_losses_231151�
&FinalLayerNorm/StatefulPartitionedCallStatefulPartitionedCall6transformer_encoder_5/StatefulPartitionedCall:output:0finallayernorm_231186finallayernorm_231188*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������P	*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_FinalLayerNorm_layer_call_and_return_conditional_losses_230477�
0ReduceStackDimensionViaSummation/PartitionedCallPartitionedCall/FinalLayerNorm/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *e
f`R^
\__inference_ReduceStackDimensionViaSummation_layer_call_and_return_conditional_losses_231198r
StandardizeTimeLimit/CastCasttimelimitinput*

DstT0*

SrcT0*'
_output_shapes
:����������
$StandardizeTimeLimit/PartitionedCallPartitionedCallStandardizeTimeLimit/Cast:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *Y
fTRR
P__inference_StandardizeTimeLimit_layer_call_and_return_conditional_losses_231208�
 ConcatenateLayer/PartitionedCallPartitionedCall9ReduceStackDimensionViaSummation/PartitionedCall:output:0-StandardizeTimeLimit/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_ConcatenateLayer_layer_call_and_return_conditional_losses_230508�
6FullyConnectedLayerImprovement/StatefulPartitionedCallStatefulPartitionedCall)ConcatenateLayer/PartitionedCall:output:0%fullyconnectedlayerimprovement_231211%fullyconnectedlayerimprovement_231213*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *c
f^R\
Z__inference_FullyConnectedLayerImprovement_layer_call_and_return_conditional_losses_230527�
-PredictionImprovement/StatefulPartitionedCallStatefulPartitionedCall?FullyConnectedLayerImprovement/StatefulPartitionedCall:output:0predictionimprovement_231216predictionimprovement_231218*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Z
fURS
Q__inference_PredictionImprovement_layer_call_and_return_conditional_losses_230543�
Output/PartitionedCallPartitionedCall6PredictionImprovement/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_Output_layer_call_and_return_conditional_losses_231225_
IdentityIdentityOutput/PartitionedCall:output:0^NoOp*
T0*
_output_shapes
:�
NoOpNoOp'^FinalLayerNorm/StatefulPartitionedCall7^FullyConnectedLayerImprovement/StatefulPartitionedCall.^PredictionImprovement/StatefulPartitionedCall.^transformer_encoder_3/StatefulPartitionedCall.^transformer_encoder_4/StatefulPartitionedCall.^transformer_encoder_5/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:���������P	:���������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2P
&FinalLayerNorm/StatefulPartitionedCall&FinalLayerNorm/StatefulPartitionedCall2p
6FullyConnectedLayerImprovement/StatefulPartitionedCall6FullyConnectedLayerImprovement/StatefulPartitionedCall2^
-PredictionImprovement/StatefulPartitionedCall-PredictionImprovement/StatefulPartitionedCall2^
-transformer_encoder_3/StatefulPartitionedCall-transformer_encoder_3/StatefulPartitionedCall2^
-transformer_encoder_4/StatefulPartitionedCall-transformer_encoder_4/StatefulPartitionedCall2^
-transformer_encoder_5/StatefulPartitionedCall-transformer_encoder_5/StatefulPartitionedCall:&7"
 
_user_specified_name231218:&6"
 
_user_specified_name231216:&5"
 
_user_specified_name231213:&4"
 
_user_specified_name231211:&3"
 
_user_specified_name231188:&2"
 
_user_specified_name231186:&1"
 
_user_specified_name231182:&0"
 
_user_specified_name231180:&/"
 
_user_specified_name231178:&."
 
_user_specified_name231176:&-"
 
_user_specified_name231174:&,"
 
_user_specified_name231172:&+"
 
_user_specified_name231170:&*"
 
_user_specified_name231168:&)"
 
_user_specified_name231166:&("
 
_user_specified_name231164:&'"
 
_user_specified_name231162:&&"
 
_user_specified_name231160:&%"
 
_user_specified_name231158:&$"
 
_user_specified_name231156:&#"
 
_user_specified_name231154:&""
 
_user_specified_name231152:&!"
 
_user_specified_name230974:& "
 
_user_specified_name230972:&"
 
_user_specified_name230970:&"
 
_user_specified_name230968:&"
 
_user_specified_name230966:&"
 
_user_specified_name230964:&"
 
_user_specified_name230962:&"
 
_user_specified_name230960:&"
 
_user_specified_name230958:&"
 
_user_specified_name230956:&"
 
_user_specified_name230954:&"
 
_user_specified_name230952:&"
 
_user_specified_name230950:&"
 
_user_specified_name230948:&"
 
_user_specified_name230946:&"
 
_user_specified_name230944:&"
 
_user_specified_name230766:&"
 
_user_specified_name230764:&"
 
_user_specified_name230762:&"
 
_user_specified_name230760:&"
 
_user_specified_name230758:&"
 
_user_specified_name230756:&"
 
_user_specified_name230754:&
"
 
_user_specified_name230752:&	"
 
_user_specified_name230750:&"
 
_user_specified_name230748:&"
 
_user_specified_name230746:&"
 
_user_specified_name230744:&"
 
_user_specified_name230742:&"
 
_user_specified_name230740:&"
 
_user_specified_name230738:&"
 
_user_specified_name230736:WS
'
_output_shapes
:���������
(
_user_specified_nameTimeLimitInput:d `
+
_output_shapes
:���������P	
1
_user_specified_nameStackLevelInputFeatures
�
]
1__inference_ConcatenateLayer_layer_call_fn_233318
inputs_0
inputs_1
identity�
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_ConcatenateLayer_layer_call_and_return_conditional_losses_230508`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:���������	:���������:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs_1:Q M
'
_output_shapes
:���������	
"
_user_specified_name
inputs_0
�
C
'__inference_Output_layer_call_fn_233377

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_Output_layer_call_and_return_conditional_losses_230553Q
IdentityIdentityPartitionedCall:output:0*
T0*
_output_shapes
:"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
6__inference_transformer_encoder_5_layer_call_fn_232847

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:
	unknown_3:	
	unknown_4:
	unknown_5:	
	unknown_6:
	unknown_7:	
	unknown_8:	
	unknown_9:	

unknown_10:	

unknown_11:		

unknown_12:	

unknown_13:		

unknown_14:	
identity

identity_1��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *F
_output_shapes4
2:���������P	:���������PP*2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *Z
fURS
Q__inference_transformer_encoder_5_layer_call_and_return_conditional_losses_231151s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������P	y

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*/
_output_shapes
:���������PP<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:���������P	: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name232841:&"
 
_user_specified_name232839:&"
 
_user_specified_name232837:&"
 
_user_specified_name232835:&"
 
_user_specified_name232833:&"
 
_user_specified_name232831:&
"
 
_user_specified_name232829:&	"
 
_user_specified_name232827:&"
 
_user_specified_name232825:&"
 
_user_specified_name232823:&"
 
_user_specified_name232821:&"
 
_user_specified_name232819:&"
 
_user_specified_name232817:&"
 
_user_specified_name232815:&"
 
_user_specified_name232813:&"
 
_user_specified_name232811:S O
+
_output_shapes
:���������P	
 
_user_specified_nameinputs
�
�
Z__inference_FullyConnectedLayerImprovement_layer_call_and_return_conditional_losses_230527

inputs0
matmul_readvariableop_resource:

-
biasadd_readvariableop_resource:

identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:

*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
O

Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?h
Gelu/mulMulGelu/mul/x:output:0BiasAdd:output:0*
T0*'
_output_shapes
:���������
P
Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *��?q
Gelu/truedivRealDivBiasAdd:output:0Gelu/Cast/x:output:0*
T0*'
_output_shapes
:���������
S
Gelu/ErfErfGelu/truediv:z:0*
T0*'
_output_shapes
:���������
O

Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?f
Gelu/addAddV2Gelu/add/x:output:0Gelu/Erf:y:0*
T0*'
_output_shapes
:���������
_

Gelu/mul_1MulGelu/mul:z:0Gelu/add:z:0*
T0*'
_output_shapes
:���������
]
IdentityIdentityGelu/mul_1:z:0^NoOp*
T0*'
_output_shapes
:���������
S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
'
_output_shapes
:���������

 
_user_specified_nameinputs
��
�
Q__inference_transformer_encoder_4_layer_call_and_return_conditional_losses_232595

inputsJ
<encoder_1st_normalizationlayer_2_mul_readvariableop_resource:	J
<encoder_1st_normalizationlayer_2_add_readvariableop_resource:	^
Hencoder_selfattentionlayer_2_query_einsum_einsum_readvariableop_resource:	P
>encoder_selfattentionlayer_2_query_add_readvariableop_resource:\
Fencoder_selfattentionlayer_2_key_einsum_einsum_readvariableop_resource:	N
<encoder_selfattentionlayer_2_key_add_readvariableop_resource:^
Hencoder_selfattentionlayer_2_value_einsum_einsum_readvariableop_resource:	P
>encoder_selfattentionlayer_2_value_add_readvariableop_resource:i
Sencoder_selfattentionlayer_2_attention_output_einsum_einsum_readvariableop_resource:	W
Iencoder_selfattentionlayer_2_attention_output_add_readvariableop_resource:	J
<encoder_2nd_normalizationlayer_2_mul_readvariableop_resource:	J
<encoder_2nd_normalizationlayer_2_add_readvariableop_resource:	P
>encoder_feedforwardlayer_1_2_tensordot_readvariableop_resource:		J
<encoder_feedforwardlayer_1_2_biasadd_readvariableop_resource:	P
>encoder_feedforwardlayer_2_2_tensordot_readvariableop_resource:		J
<encoder_feedforwardlayer_2_2_biasadd_readvariableop_resource:	
identity

identity_1��3Encoder-1st-NormalizationLayer-2/add/ReadVariableOp�3Encoder-1st-NormalizationLayer-2/mul/ReadVariableOp�3Encoder-2nd-NormalizationLayer-2/add/ReadVariableOp�3Encoder-2nd-NormalizationLayer-2/mul/ReadVariableOp�3Encoder-FeedForwardLayer_1_2/BiasAdd/ReadVariableOp�5Encoder-FeedForwardLayer_1_2/Tensordot/ReadVariableOp�3Encoder-FeedForwardLayer_2_2/BiasAdd/ReadVariableOp�5Encoder-FeedForwardLayer_2_2/Tensordot/ReadVariableOp�@Encoder-SelfAttentionLayer-2/attention_output/add/ReadVariableOp�JEncoder-SelfAttentionLayer-2/attention_output/einsum/Einsum/ReadVariableOp�3Encoder-SelfAttentionLayer-2/key/add/ReadVariableOp�=Encoder-SelfAttentionLayer-2/key/einsum/Einsum/ReadVariableOp�5Encoder-SelfAttentionLayer-2/query/add/ReadVariableOp�?Encoder-SelfAttentionLayer-2/query/einsum/Einsum/ReadVariableOp�5Encoder-SelfAttentionLayer-2/value/add/ReadVariableOp�?Encoder-SelfAttentionLayer-2/value/einsum/Einsum/ReadVariableOpj
&Encoder-1st-NormalizationLayer-2/ShapeShapeinputs*
T0*
_output_shapes
::��~
4Encoder-1st-NormalizationLayer-2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
6Encoder-1st-NormalizationLayer-2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
6Encoder-1st-NormalizationLayer-2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
.Encoder-1st-NormalizationLayer-2/strided_sliceStridedSlice/Encoder-1st-NormalizationLayer-2/Shape:output:0=Encoder-1st-NormalizationLayer-2/strided_slice/stack:output:0?Encoder-1st-NormalizationLayer-2/strided_slice/stack_1:output:0?Encoder-1st-NormalizationLayer-2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskp
&Encoder-1st-NormalizationLayer-2/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
%Encoder-1st-NormalizationLayer-2/ProdProd7Encoder-1st-NormalizationLayer-2/strided_slice:output:0/Encoder-1st-NormalizationLayer-2/Const:output:0*
T0*
_output_shapes
: �
6Encoder-1st-NormalizationLayer-2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:�
8Encoder-1st-NormalizationLayer-2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: �
8Encoder-1st-NormalizationLayer-2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
0Encoder-1st-NormalizationLayer-2/strided_slice_1StridedSlice/Encoder-1st-NormalizationLayer-2/Shape:output:0?Encoder-1st-NormalizationLayer-2/strided_slice_1/stack:output:0AEncoder-1st-NormalizationLayer-2/strided_slice_1/stack_1:output:0AEncoder-1st-NormalizationLayer-2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_maskr
(Encoder-1st-NormalizationLayer-2/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
'Encoder-1st-NormalizationLayer-2/Prod_1Prod9Encoder-1st-NormalizationLayer-2/strided_slice_1:output:01Encoder-1st-NormalizationLayer-2/Const_1:output:0*
T0*
_output_shapes
: r
0Encoder-1st-NormalizationLayer-2/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :r
0Encoder-1st-NormalizationLayer-2/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :�
.Encoder-1st-NormalizationLayer-2/Reshape/shapePack9Encoder-1st-NormalizationLayer-2/Reshape/shape/0:output:0.Encoder-1st-NormalizationLayer-2/Prod:output:00Encoder-1st-NormalizationLayer-2/Prod_1:output:09Encoder-1st-NormalizationLayer-2/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:�
(Encoder-1st-NormalizationLayer-2/ReshapeReshapeinputs7Encoder-1st-NormalizationLayer-2/Reshape/shape:output:0*
T0*8
_output_shapes&
$:"�������������������
,Encoder-1st-NormalizationLayer-2/ones/packedPack.Encoder-1st-NormalizationLayer-2/Prod:output:0*
N*
T0*
_output_shapes
:p
+Encoder-1st-NormalizationLayer-2/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
%Encoder-1st-NormalizationLayer-2/onesFill5Encoder-1st-NormalizationLayer-2/ones/packed:output:04Encoder-1st-NormalizationLayer-2/ones/Const:output:0*
T0*#
_output_shapes
:����������
-Encoder-1st-NormalizationLayer-2/zeros/packedPack.Encoder-1st-NormalizationLayer-2/Prod:output:0*
N*
T0*
_output_shapes
:q
,Encoder-1st-NormalizationLayer-2/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
&Encoder-1st-NormalizationLayer-2/zerosFill6Encoder-1st-NormalizationLayer-2/zeros/packed:output:05Encoder-1st-NormalizationLayer-2/zeros/Const:output:0*
T0*#
_output_shapes
:���������k
(Encoder-1st-NormalizationLayer-2/Const_2Const*
_output_shapes
: *
dtype0*
valueB k
(Encoder-1st-NormalizationLayer-2/Const_3Const*
_output_shapes
: *
dtype0*
valueB �
1Encoder-1st-NormalizationLayer-2/FusedBatchNormV3FusedBatchNormV31Encoder-1st-NormalizationLayer-2/Reshape:output:0.Encoder-1st-NormalizationLayer-2/ones:output:0/Encoder-1st-NormalizationLayer-2/zeros:output:01Encoder-1st-NormalizationLayer-2/Const_2:output:01Encoder-1st-NormalizationLayer-2/Const_3:output:0*
T0*
U0*x
_output_shapesf
d:"������������������:���������:���������:���������:���������:*
data_formatNCHW*
epsilon%o�:�
*Encoder-1st-NormalizationLayer-2/Reshape_1Reshape5Encoder-1st-NormalizationLayer-2/FusedBatchNormV3:y:0/Encoder-1st-NormalizationLayer-2/Shape:output:0*
T0*+
_output_shapes
:���������P	�
3Encoder-1st-NormalizationLayer-2/mul/ReadVariableOpReadVariableOp<encoder_1st_normalizationlayer_2_mul_readvariableop_resource*
_output_shapes
:	*
dtype0�
$Encoder-1st-NormalizationLayer-2/mulMul3Encoder-1st-NormalizationLayer-2/Reshape_1:output:0;Encoder-1st-NormalizationLayer-2/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������P	�
3Encoder-1st-NormalizationLayer-2/add/ReadVariableOpReadVariableOp<encoder_1st_normalizationlayer_2_add_readvariableop_resource*
_output_shapes
:	*
dtype0�
$Encoder-1st-NormalizationLayer-2/addAddV2(Encoder-1st-NormalizationLayer-2/mul:z:0;Encoder-1st-NormalizationLayer-2/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������P	�
?Encoder-SelfAttentionLayer-2/query/einsum/Einsum/ReadVariableOpReadVariableOpHencoder_selfattentionlayer_2_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:	*
dtype0�
0Encoder-SelfAttentionLayer-2/query/einsum/EinsumEinsum(Encoder-1st-NormalizationLayer-2/add:z:0GEncoder-SelfAttentionLayer-2/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������P*
equationabc,cde->abde�
5Encoder-SelfAttentionLayer-2/query/add/ReadVariableOpReadVariableOp>encoder_selfattentionlayer_2_query_add_readvariableop_resource*
_output_shapes

:*
dtype0�
&Encoder-SelfAttentionLayer-2/query/addAddV29Encoder-SelfAttentionLayer-2/query/einsum/Einsum:output:0=Encoder-SelfAttentionLayer-2/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������P�
=Encoder-SelfAttentionLayer-2/key/einsum/Einsum/ReadVariableOpReadVariableOpFencoder_selfattentionlayer_2_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:	*
dtype0�
.Encoder-SelfAttentionLayer-2/key/einsum/EinsumEinsum(Encoder-1st-NormalizationLayer-2/add:z:0EEncoder-SelfAttentionLayer-2/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������P*
equationabc,cde->abde�
3Encoder-SelfAttentionLayer-2/key/add/ReadVariableOpReadVariableOp<encoder_selfattentionlayer_2_key_add_readvariableop_resource*
_output_shapes

:*
dtype0�
$Encoder-SelfAttentionLayer-2/key/addAddV27Encoder-SelfAttentionLayer-2/key/einsum/Einsum:output:0;Encoder-SelfAttentionLayer-2/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������P�
?Encoder-SelfAttentionLayer-2/value/einsum/Einsum/ReadVariableOpReadVariableOpHencoder_selfattentionlayer_2_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:	*
dtype0�
0Encoder-SelfAttentionLayer-2/value/einsum/EinsumEinsum(Encoder-1st-NormalizationLayer-2/add:z:0GEncoder-SelfAttentionLayer-2/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������P*
equationabc,cde->abde�
5Encoder-SelfAttentionLayer-2/value/add/ReadVariableOpReadVariableOp>encoder_selfattentionlayer_2_value_add_readvariableop_resource*
_output_shapes

:*
dtype0�
&Encoder-SelfAttentionLayer-2/value/addAddV29Encoder-SelfAttentionLayer-2/value/einsum/Einsum:output:0=Encoder-SelfAttentionLayer-2/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������Pg
"Encoder-SelfAttentionLayer-2/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *:�?�
 Encoder-SelfAttentionLayer-2/MulMul*Encoder-SelfAttentionLayer-2/query/add:z:0+Encoder-SelfAttentionLayer-2/Mul/y:output:0*
T0*/
_output_shapes
:���������P�
*Encoder-SelfAttentionLayer-2/einsum/EinsumEinsum(Encoder-SelfAttentionLayer-2/key/add:z:0$Encoder-SelfAttentionLayer-2/Mul:z:0*
N*
T0*/
_output_shapes
:���������PP*
equationaecd,abcd->acbe�
,Encoder-SelfAttentionLayer-2/softmax/SoftmaxSoftmax3Encoder-SelfAttentionLayer-2/einsum/Einsum:output:0*
T0*/
_output_shapes
:���������PPw
2Encoder-SelfAttentionLayer-2/dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?�
0Encoder-SelfAttentionLayer-2/dropout/dropout/MulMul6Encoder-SelfAttentionLayer-2/softmax/Softmax:softmax:0;Encoder-SelfAttentionLayer-2/dropout/dropout/Const:output:0*
T0*/
_output_shapes
:���������PP�
2Encoder-SelfAttentionLayer-2/dropout/dropout/ShapeShape6Encoder-SelfAttentionLayer-2/softmax/Softmax:softmax:0*
T0*
_output_shapes
::���
IEncoder-SelfAttentionLayer-2/dropout/dropout/random_uniform/RandomUniformRandomUniform;Encoder-SelfAttentionLayer-2/dropout/dropout/Shape:output:0*
T0*/
_output_shapes
:���������PP*
dtype0*
seed���
;Encoder-SelfAttentionLayer-2/dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=�
9Encoder-SelfAttentionLayer-2/dropout/dropout/GreaterEqualGreaterEqualREncoder-SelfAttentionLayer-2/dropout/dropout/random_uniform/RandomUniform:output:0DEncoder-SelfAttentionLayer-2/dropout/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:���������PPy
4Encoder-SelfAttentionLayer-2/dropout/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
5Encoder-SelfAttentionLayer-2/dropout/dropout/SelectV2SelectV2=Encoder-SelfAttentionLayer-2/dropout/dropout/GreaterEqual:z:04Encoder-SelfAttentionLayer-2/dropout/dropout/Mul:z:0=Encoder-SelfAttentionLayer-2/dropout/dropout/Const_1:output:0*
T0*/
_output_shapes
:���������PP�
,Encoder-SelfAttentionLayer-2/einsum_1/EinsumEinsum>Encoder-SelfAttentionLayer-2/dropout/dropout/SelectV2:output:0*Encoder-SelfAttentionLayer-2/value/add:z:0*
N*
T0*/
_output_shapes
:���������P*
equationacbe,aecd->abcd�
JEncoder-SelfAttentionLayer-2/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpSencoder_selfattentionlayer_2_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:	*
dtype0�
;Encoder-SelfAttentionLayer-2/attention_output/einsum/EinsumEinsum5Encoder-SelfAttentionLayer-2/einsum_1/Einsum:output:0REncoder-SelfAttentionLayer-2/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:���������P	*
equationabcd,cde->abe�
@Encoder-SelfAttentionLayer-2/attention_output/add/ReadVariableOpReadVariableOpIencoder_selfattentionlayer_2_attention_output_add_readvariableop_resource*
_output_shapes
:	*
dtype0�
1Encoder-SelfAttentionLayer-2/attention_output/addAddV2DEncoder-SelfAttentionLayer-2/attention_output/einsum/Einsum:output:0HEncoder-SelfAttentionLayer-2/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������P	�
Encoder-1st-AdditionLayer-2/addAddV2inputs5Encoder-SelfAttentionLayer-2/attention_output/add:z:0*
T0*+
_output_shapes
:���������P	�
&Encoder-2nd-NormalizationLayer-2/ShapeShape#Encoder-1st-AdditionLayer-2/add:z:0*
T0*
_output_shapes
::��~
4Encoder-2nd-NormalizationLayer-2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
6Encoder-2nd-NormalizationLayer-2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
6Encoder-2nd-NormalizationLayer-2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
.Encoder-2nd-NormalizationLayer-2/strided_sliceStridedSlice/Encoder-2nd-NormalizationLayer-2/Shape:output:0=Encoder-2nd-NormalizationLayer-2/strided_slice/stack:output:0?Encoder-2nd-NormalizationLayer-2/strided_slice/stack_1:output:0?Encoder-2nd-NormalizationLayer-2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskp
&Encoder-2nd-NormalizationLayer-2/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
%Encoder-2nd-NormalizationLayer-2/ProdProd7Encoder-2nd-NormalizationLayer-2/strided_slice:output:0/Encoder-2nd-NormalizationLayer-2/Const:output:0*
T0*
_output_shapes
: �
6Encoder-2nd-NormalizationLayer-2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:�
8Encoder-2nd-NormalizationLayer-2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: �
8Encoder-2nd-NormalizationLayer-2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
0Encoder-2nd-NormalizationLayer-2/strided_slice_1StridedSlice/Encoder-2nd-NormalizationLayer-2/Shape:output:0?Encoder-2nd-NormalizationLayer-2/strided_slice_1/stack:output:0AEncoder-2nd-NormalizationLayer-2/strided_slice_1/stack_1:output:0AEncoder-2nd-NormalizationLayer-2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_maskr
(Encoder-2nd-NormalizationLayer-2/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
'Encoder-2nd-NormalizationLayer-2/Prod_1Prod9Encoder-2nd-NormalizationLayer-2/strided_slice_1:output:01Encoder-2nd-NormalizationLayer-2/Const_1:output:0*
T0*
_output_shapes
: r
0Encoder-2nd-NormalizationLayer-2/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :r
0Encoder-2nd-NormalizationLayer-2/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :�
.Encoder-2nd-NormalizationLayer-2/Reshape/shapePack9Encoder-2nd-NormalizationLayer-2/Reshape/shape/0:output:0.Encoder-2nd-NormalizationLayer-2/Prod:output:00Encoder-2nd-NormalizationLayer-2/Prod_1:output:09Encoder-2nd-NormalizationLayer-2/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:�
(Encoder-2nd-NormalizationLayer-2/ReshapeReshape#Encoder-1st-AdditionLayer-2/add:z:07Encoder-2nd-NormalizationLayer-2/Reshape/shape:output:0*
T0*8
_output_shapes&
$:"�������������������
,Encoder-2nd-NormalizationLayer-2/ones/packedPack.Encoder-2nd-NormalizationLayer-2/Prod:output:0*
N*
T0*
_output_shapes
:p
+Encoder-2nd-NormalizationLayer-2/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
%Encoder-2nd-NormalizationLayer-2/onesFill5Encoder-2nd-NormalizationLayer-2/ones/packed:output:04Encoder-2nd-NormalizationLayer-2/ones/Const:output:0*
T0*#
_output_shapes
:����������
-Encoder-2nd-NormalizationLayer-2/zeros/packedPack.Encoder-2nd-NormalizationLayer-2/Prod:output:0*
N*
T0*
_output_shapes
:q
,Encoder-2nd-NormalizationLayer-2/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
&Encoder-2nd-NormalizationLayer-2/zerosFill6Encoder-2nd-NormalizationLayer-2/zeros/packed:output:05Encoder-2nd-NormalizationLayer-2/zeros/Const:output:0*
T0*#
_output_shapes
:���������k
(Encoder-2nd-NormalizationLayer-2/Const_2Const*
_output_shapes
: *
dtype0*
valueB k
(Encoder-2nd-NormalizationLayer-2/Const_3Const*
_output_shapes
: *
dtype0*
valueB �
1Encoder-2nd-NormalizationLayer-2/FusedBatchNormV3FusedBatchNormV31Encoder-2nd-NormalizationLayer-2/Reshape:output:0.Encoder-2nd-NormalizationLayer-2/ones:output:0/Encoder-2nd-NormalizationLayer-2/zeros:output:01Encoder-2nd-NormalizationLayer-2/Const_2:output:01Encoder-2nd-NormalizationLayer-2/Const_3:output:0*
T0*
U0*x
_output_shapesf
d:"������������������:���������:���������:���������:���������:*
data_formatNCHW*
epsilon%o�:�
*Encoder-2nd-NormalizationLayer-2/Reshape_1Reshape5Encoder-2nd-NormalizationLayer-2/FusedBatchNormV3:y:0/Encoder-2nd-NormalizationLayer-2/Shape:output:0*
T0*+
_output_shapes
:���������P	�
3Encoder-2nd-NormalizationLayer-2/mul/ReadVariableOpReadVariableOp<encoder_2nd_normalizationlayer_2_mul_readvariableop_resource*
_output_shapes
:	*
dtype0�
$Encoder-2nd-NormalizationLayer-2/mulMul3Encoder-2nd-NormalizationLayer-2/Reshape_1:output:0;Encoder-2nd-NormalizationLayer-2/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������P	�
3Encoder-2nd-NormalizationLayer-2/add/ReadVariableOpReadVariableOp<encoder_2nd_normalizationlayer_2_add_readvariableop_resource*
_output_shapes
:	*
dtype0�
$Encoder-2nd-NormalizationLayer-2/addAddV2(Encoder-2nd-NormalizationLayer-2/mul:z:0;Encoder-2nd-NormalizationLayer-2/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������P	�
5Encoder-FeedForwardLayer_1_2/Tensordot/ReadVariableOpReadVariableOp>encoder_feedforwardlayer_1_2_tensordot_readvariableop_resource*
_output_shapes

:		*
dtype0u
+Encoder-FeedForwardLayer_1_2/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:|
+Encoder-FeedForwardLayer_1_2/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       �
,Encoder-FeedForwardLayer_1_2/Tensordot/ShapeShape(Encoder-2nd-NormalizationLayer-2/add:z:0*
T0*
_output_shapes
::��v
4Encoder-FeedForwardLayer_1_2/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
/Encoder-FeedForwardLayer_1_2/Tensordot/GatherV2GatherV25Encoder-FeedForwardLayer_1_2/Tensordot/Shape:output:04Encoder-FeedForwardLayer_1_2/Tensordot/free:output:0=Encoder-FeedForwardLayer_1_2/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:x
6Encoder-FeedForwardLayer_1_2/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
1Encoder-FeedForwardLayer_1_2/Tensordot/GatherV2_1GatherV25Encoder-FeedForwardLayer_1_2/Tensordot/Shape:output:04Encoder-FeedForwardLayer_1_2/Tensordot/axes:output:0?Encoder-FeedForwardLayer_1_2/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:v
,Encoder-FeedForwardLayer_1_2/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
+Encoder-FeedForwardLayer_1_2/Tensordot/ProdProd8Encoder-FeedForwardLayer_1_2/Tensordot/GatherV2:output:05Encoder-FeedForwardLayer_1_2/Tensordot/Const:output:0*
T0*
_output_shapes
: x
.Encoder-FeedForwardLayer_1_2/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
-Encoder-FeedForwardLayer_1_2/Tensordot/Prod_1Prod:Encoder-FeedForwardLayer_1_2/Tensordot/GatherV2_1:output:07Encoder-FeedForwardLayer_1_2/Tensordot/Const_1:output:0*
T0*
_output_shapes
: t
2Encoder-FeedForwardLayer_1_2/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
-Encoder-FeedForwardLayer_1_2/Tensordot/concatConcatV24Encoder-FeedForwardLayer_1_2/Tensordot/free:output:04Encoder-FeedForwardLayer_1_2/Tensordot/axes:output:0;Encoder-FeedForwardLayer_1_2/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
,Encoder-FeedForwardLayer_1_2/Tensordot/stackPack4Encoder-FeedForwardLayer_1_2/Tensordot/Prod:output:06Encoder-FeedForwardLayer_1_2/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
0Encoder-FeedForwardLayer_1_2/Tensordot/transpose	Transpose(Encoder-2nd-NormalizationLayer-2/add:z:06Encoder-FeedForwardLayer_1_2/Tensordot/concat:output:0*
T0*+
_output_shapes
:���������P	�
.Encoder-FeedForwardLayer_1_2/Tensordot/ReshapeReshape4Encoder-FeedForwardLayer_1_2/Tensordot/transpose:y:05Encoder-FeedForwardLayer_1_2/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
-Encoder-FeedForwardLayer_1_2/Tensordot/MatMulMatMul7Encoder-FeedForwardLayer_1_2/Tensordot/Reshape:output:0=Encoder-FeedForwardLayer_1_2/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������	x
.Encoder-FeedForwardLayer_1_2/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:	v
4Encoder-FeedForwardLayer_1_2/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
/Encoder-FeedForwardLayer_1_2/Tensordot/concat_1ConcatV28Encoder-FeedForwardLayer_1_2/Tensordot/GatherV2:output:07Encoder-FeedForwardLayer_1_2/Tensordot/Const_2:output:0=Encoder-FeedForwardLayer_1_2/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
&Encoder-FeedForwardLayer_1_2/TensordotReshape7Encoder-FeedForwardLayer_1_2/Tensordot/MatMul:product:08Encoder-FeedForwardLayer_1_2/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������P	�
3Encoder-FeedForwardLayer_1_2/BiasAdd/ReadVariableOpReadVariableOp<encoder_feedforwardlayer_1_2_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype0�
$Encoder-FeedForwardLayer_1_2/BiasAddBiasAdd/Encoder-FeedForwardLayer_1_2/Tensordot:output:0;Encoder-FeedForwardLayer_1_2/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������P	l
'Encoder-FeedForwardLayer_1_2/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
%Encoder-FeedForwardLayer_1_2/Gelu/mulMul0Encoder-FeedForwardLayer_1_2/Gelu/mul/x:output:0-Encoder-FeedForwardLayer_1_2/BiasAdd:output:0*
T0*+
_output_shapes
:���������P	m
(Encoder-FeedForwardLayer_1_2/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *��?�
)Encoder-FeedForwardLayer_1_2/Gelu/truedivRealDiv-Encoder-FeedForwardLayer_1_2/BiasAdd:output:01Encoder-FeedForwardLayer_1_2/Gelu/Cast/x:output:0*
T0*+
_output_shapes
:���������P	�
%Encoder-FeedForwardLayer_1_2/Gelu/ErfErf-Encoder-FeedForwardLayer_1_2/Gelu/truediv:z:0*
T0*+
_output_shapes
:���������P	l
'Encoder-FeedForwardLayer_1_2/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
%Encoder-FeedForwardLayer_1_2/Gelu/addAddV20Encoder-FeedForwardLayer_1_2/Gelu/add/x:output:0)Encoder-FeedForwardLayer_1_2/Gelu/Erf:y:0*
T0*+
_output_shapes
:���������P	�
'Encoder-FeedForwardLayer_1_2/Gelu/mul_1Mul)Encoder-FeedForwardLayer_1_2/Gelu/mul:z:0)Encoder-FeedForwardLayer_1_2/Gelu/add:z:0*
T0*+
_output_shapes
:���������P	�
5Encoder-FeedForwardLayer_2_2/Tensordot/ReadVariableOpReadVariableOp>encoder_feedforwardlayer_2_2_tensordot_readvariableop_resource*
_output_shapes

:		*
dtype0u
+Encoder-FeedForwardLayer_2_2/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:|
+Encoder-FeedForwardLayer_2_2/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       �
,Encoder-FeedForwardLayer_2_2/Tensordot/ShapeShape+Encoder-FeedForwardLayer_1_2/Gelu/mul_1:z:0*
T0*
_output_shapes
::��v
4Encoder-FeedForwardLayer_2_2/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
/Encoder-FeedForwardLayer_2_2/Tensordot/GatherV2GatherV25Encoder-FeedForwardLayer_2_2/Tensordot/Shape:output:04Encoder-FeedForwardLayer_2_2/Tensordot/free:output:0=Encoder-FeedForwardLayer_2_2/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:x
6Encoder-FeedForwardLayer_2_2/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
1Encoder-FeedForwardLayer_2_2/Tensordot/GatherV2_1GatherV25Encoder-FeedForwardLayer_2_2/Tensordot/Shape:output:04Encoder-FeedForwardLayer_2_2/Tensordot/axes:output:0?Encoder-FeedForwardLayer_2_2/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:v
,Encoder-FeedForwardLayer_2_2/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
+Encoder-FeedForwardLayer_2_2/Tensordot/ProdProd8Encoder-FeedForwardLayer_2_2/Tensordot/GatherV2:output:05Encoder-FeedForwardLayer_2_2/Tensordot/Const:output:0*
T0*
_output_shapes
: x
.Encoder-FeedForwardLayer_2_2/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
-Encoder-FeedForwardLayer_2_2/Tensordot/Prod_1Prod:Encoder-FeedForwardLayer_2_2/Tensordot/GatherV2_1:output:07Encoder-FeedForwardLayer_2_2/Tensordot/Const_1:output:0*
T0*
_output_shapes
: t
2Encoder-FeedForwardLayer_2_2/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
-Encoder-FeedForwardLayer_2_2/Tensordot/concatConcatV24Encoder-FeedForwardLayer_2_2/Tensordot/free:output:04Encoder-FeedForwardLayer_2_2/Tensordot/axes:output:0;Encoder-FeedForwardLayer_2_2/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
,Encoder-FeedForwardLayer_2_2/Tensordot/stackPack4Encoder-FeedForwardLayer_2_2/Tensordot/Prod:output:06Encoder-FeedForwardLayer_2_2/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
0Encoder-FeedForwardLayer_2_2/Tensordot/transpose	Transpose+Encoder-FeedForwardLayer_1_2/Gelu/mul_1:z:06Encoder-FeedForwardLayer_2_2/Tensordot/concat:output:0*
T0*+
_output_shapes
:���������P	�
.Encoder-FeedForwardLayer_2_2/Tensordot/ReshapeReshape4Encoder-FeedForwardLayer_2_2/Tensordot/transpose:y:05Encoder-FeedForwardLayer_2_2/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
-Encoder-FeedForwardLayer_2_2/Tensordot/MatMulMatMul7Encoder-FeedForwardLayer_2_2/Tensordot/Reshape:output:0=Encoder-FeedForwardLayer_2_2/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������	x
.Encoder-FeedForwardLayer_2_2/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:	v
4Encoder-FeedForwardLayer_2_2/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
/Encoder-FeedForwardLayer_2_2/Tensordot/concat_1ConcatV28Encoder-FeedForwardLayer_2_2/Tensordot/GatherV2:output:07Encoder-FeedForwardLayer_2_2/Tensordot/Const_2:output:0=Encoder-FeedForwardLayer_2_2/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
&Encoder-FeedForwardLayer_2_2/TensordotReshape7Encoder-FeedForwardLayer_2_2/Tensordot/MatMul:product:08Encoder-FeedForwardLayer_2_2/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������P	�
3Encoder-FeedForwardLayer_2_2/BiasAdd/ReadVariableOpReadVariableOp<encoder_feedforwardlayer_2_2_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype0�
$Encoder-FeedForwardLayer_2_2/BiasAddBiasAdd/Encoder-FeedForwardLayer_2_2/Tensordot:output:0;Encoder-FeedForwardLayer_2_2/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������P	\
dropout_4/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?�
dropout_4/dropout/MulMul-Encoder-FeedForwardLayer_2_2/BiasAdd:output:0 dropout_4/dropout/Const:output:0*
T0*+
_output_shapes
:���������P	�
dropout_4/dropout/ShapeShape-Encoder-FeedForwardLayer_2_2/BiasAdd:output:0*
T0*
_output_shapes
::���
.dropout_4/dropout/random_uniform/RandomUniformRandomUniform dropout_4/dropout/Shape:output:0*
T0*+
_output_shapes
:���������P	*
dtype0*
seed2*
seed��e
 dropout_4/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=�
dropout_4/dropout/GreaterEqualGreaterEqual7dropout_4/dropout/random_uniform/RandomUniform:output:0)dropout_4/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:���������P	^
dropout_4/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout_4/dropout/SelectV2SelectV2"dropout_4/dropout/GreaterEqual:z:0dropout_4/dropout/Mul:z:0"dropout_4/dropout/Const_1:output:0*
T0*+
_output_shapes
:���������P	�
Encoder-2nd-AdditionLayer-2/addAddV2#Encoder-1st-AdditionLayer-2/add:z:0#dropout_4/dropout/SelectV2:output:0*
T0*+
_output_shapes
:���������P	v
IdentityIdentity#Encoder-2nd-AdditionLayer-2/add:z:0^NoOp*
T0*+
_output_shapes
:���������P	�

Identity_1Identity6Encoder-SelfAttentionLayer-2/softmax/Softmax:softmax:0^NoOp*
T0*/
_output_shapes
:���������PP�
NoOpNoOp4^Encoder-1st-NormalizationLayer-2/add/ReadVariableOp4^Encoder-1st-NormalizationLayer-2/mul/ReadVariableOp4^Encoder-2nd-NormalizationLayer-2/add/ReadVariableOp4^Encoder-2nd-NormalizationLayer-2/mul/ReadVariableOp4^Encoder-FeedForwardLayer_1_2/BiasAdd/ReadVariableOp6^Encoder-FeedForwardLayer_1_2/Tensordot/ReadVariableOp4^Encoder-FeedForwardLayer_2_2/BiasAdd/ReadVariableOp6^Encoder-FeedForwardLayer_2_2/Tensordot/ReadVariableOpA^Encoder-SelfAttentionLayer-2/attention_output/add/ReadVariableOpK^Encoder-SelfAttentionLayer-2/attention_output/einsum/Einsum/ReadVariableOp4^Encoder-SelfAttentionLayer-2/key/add/ReadVariableOp>^Encoder-SelfAttentionLayer-2/key/einsum/Einsum/ReadVariableOp6^Encoder-SelfAttentionLayer-2/query/add/ReadVariableOp@^Encoder-SelfAttentionLayer-2/query/einsum/Einsum/ReadVariableOp6^Encoder-SelfAttentionLayer-2/value/add/ReadVariableOp@^Encoder-SelfAttentionLayer-2/value/einsum/Einsum/ReadVariableOp*
_output_shapes
 "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:���������P	: : : : : : : : : : : : : : : : 2j
3Encoder-1st-NormalizationLayer-2/add/ReadVariableOp3Encoder-1st-NormalizationLayer-2/add/ReadVariableOp2j
3Encoder-1st-NormalizationLayer-2/mul/ReadVariableOp3Encoder-1st-NormalizationLayer-2/mul/ReadVariableOp2j
3Encoder-2nd-NormalizationLayer-2/add/ReadVariableOp3Encoder-2nd-NormalizationLayer-2/add/ReadVariableOp2j
3Encoder-2nd-NormalizationLayer-2/mul/ReadVariableOp3Encoder-2nd-NormalizationLayer-2/mul/ReadVariableOp2j
3Encoder-FeedForwardLayer_1_2/BiasAdd/ReadVariableOp3Encoder-FeedForwardLayer_1_2/BiasAdd/ReadVariableOp2n
5Encoder-FeedForwardLayer_1_2/Tensordot/ReadVariableOp5Encoder-FeedForwardLayer_1_2/Tensordot/ReadVariableOp2j
3Encoder-FeedForwardLayer_2_2/BiasAdd/ReadVariableOp3Encoder-FeedForwardLayer_2_2/BiasAdd/ReadVariableOp2n
5Encoder-FeedForwardLayer_2_2/Tensordot/ReadVariableOp5Encoder-FeedForwardLayer_2_2/Tensordot/ReadVariableOp2�
@Encoder-SelfAttentionLayer-2/attention_output/add/ReadVariableOp@Encoder-SelfAttentionLayer-2/attention_output/add/ReadVariableOp2�
JEncoder-SelfAttentionLayer-2/attention_output/einsum/Einsum/ReadVariableOpJEncoder-SelfAttentionLayer-2/attention_output/einsum/Einsum/ReadVariableOp2j
3Encoder-SelfAttentionLayer-2/key/add/ReadVariableOp3Encoder-SelfAttentionLayer-2/key/add/ReadVariableOp2~
=Encoder-SelfAttentionLayer-2/key/einsum/Einsum/ReadVariableOp=Encoder-SelfAttentionLayer-2/key/einsum/Einsum/ReadVariableOp2n
5Encoder-SelfAttentionLayer-2/query/add/ReadVariableOp5Encoder-SelfAttentionLayer-2/query/add/ReadVariableOp2�
?Encoder-SelfAttentionLayer-2/query/einsum/Einsum/ReadVariableOp?Encoder-SelfAttentionLayer-2/query/einsum/Einsum/ReadVariableOp2n
5Encoder-SelfAttentionLayer-2/value/add/ReadVariableOp5Encoder-SelfAttentionLayer-2/value/add/ReadVariableOp2�
?Encoder-SelfAttentionLayer-2/value/einsum/Einsum/ReadVariableOp?Encoder-SelfAttentionLayer-2/value/einsum/Einsum/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:(
$
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:S O
+
_output_shapes
:���������P	
 
_user_specified_nameinputs
�
�
6__inference_PredictionImprovement_layer_call_fn_233361

inputs
unknown:

	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Z
fURS
Q__inference_PredictionImprovement_layer_call_and_return_conditional_losses_230543o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������
: : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name233357:&"
 
_user_specified_name233355:O K
'
_output_shapes
:���������

 
_user_specified_nameinputs
�
�
6__inference_transformer_encoder_5_layer_call_fn_232808

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:
	unknown_3:	
	unknown_4:
	unknown_5:	
	unknown_6:
	unknown_7:	
	unknown_8:	
	unknown_9:	

unknown_10:	

unknown_11:		

unknown_12:	

unknown_13:		

unknown_14:	
identity

identity_1��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *F
_output_shapes4
2:���������P	:���������PP*2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *Z
fURS
Q__inference_transformer_encoder_5_layer_call_and_return_conditional_losses_230401s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������P	y

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*/
_output_shapes
:���������PP<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:���������P	: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name232802:&"
 
_user_specified_name232800:&"
 
_user_specified_name232798:&"
 
_user_specified_name232796:&"
 
_user_specified_name232794:&"
 
_user_specified_name232792:&
"
 
_user_specified_name232790:&	"
 
_user_specified_name232788:&"
 
_user_specified_name232786:&"
 
_user_specified_name232784:&"
 
_user_specified_name232782:&"
 
_user_specified_name232780:&"
 
_user_specified_name232778:&"
 
_user_specified_name232776:&"
 
_user_specified_name232774:&"
 
_user_specified_name232772:S O
+
_output_shapes
:���������P	
 
_user_specified_nameinputs
�+
�
(__inference_model_1_layer_call_fn_231456
stacklevelinputfeatures
timelimitinput
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:
	unknown_3:	
	unknown_4:
	unknown_5:	
	unknown_6:
	unknown_7:	
	unknown_8:	
	unknown_9:	

unknown_10:	

unknown_11:		

unknown_12:	

unknown_13:		

unknown_14:	

unknown_15:	

unknown_16:	 

unknown_17:	

unknown_18: 

unknown_19:	

unknown_20: 

unknown_21:	

unknown_22: 

unknown_23:	

unknown_24:	

unknown_25:	

unknown_26:	

unknown_27:		

unknown_28:	

unknown_29:		

unknown_30:	

unknown_31:	

unknown_32:	 

unknown_33:	

unknown_34: 

unknown_35:	

unknown_36: 

unknown_37:	

unknown_38: 

unknown_39:	

unknown_40:	

unknown_41:	

unknown_42:	

unknown_43:		

unknown_44:	

unknown_45:		

unknown_46:	

unknown_47:	

unknown_48:	

unknown_49:



unknown_50:


unknown_51:


unknown_52:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallstacklevelinputfeaturestimelimitinputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50
unknown_51
unknown_52*C
Tin<
:28*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:*X
_read_only_resource_inputs:
86	
 !"#$%&'()*+,-./01234567*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_model_1_layer_call_and_return_conditional_losses_231228`
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes
:<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:���������P	:���������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&7"
 
_user_specified_name231452:&6"
 
_user_specified_name231450:&5"
 
_user_specified_name231448:&4"
 
_user_specified_name231446:&3"
 
_user_specified_name231444:&2"
 
_user_specified_name231442:&1"
 
_user_specified_name231440:&0"
 
_user_specified_name231438:&/"
 
_user_specified_name231436:&."
 
_user_specified_name231434:&-"
 
_user_specified_name231432:&,"
 
_user_specified_name231430:&+"
 
_user_specified_name231428:&*"
 
_user_specified_name231426:&)"
 
_user_specified_name231424:&("
 
_user_specified_name231422:&'"
 
_user_specified_name231420:&&"
 
_user_specified_name231418:&%"
 
_user_specified_name231416:&$"
 
_user_specified_name231414:&#"
 
_user_specified_name231412:&""
 
_user_specified_name231410:&!"
 
_user_specified_name231408:& "
 
_user_specified_name231406:&"
 
_user_specified_name231404:&"
 
_user_specified_name231402:&"
 
_user_specified_name231400:&"
 
_user_specified_name231398:&"
 
_user_specified_name231396:&"
 
_user_specified_name231394:&"
 
_user_specified_name231392:&"
 
_user_specified_name231390:&"
 
_user_specified_name231388:&"
 
_user_specified_name231386:&"
 
_user_specified_name231384:&"
 
_user_specified_name231382:&"
 
_user_specified_name231380:&"
 
_user_specified_name231378:&"
 
_user_specified_name231376:&"
 
_user_specified_name231374:&"
 
_user_specified_name231372:&"
 
_user_specified_name231370:&"
 
_user_specified_name231368:&"
 
_user_specified_name231366:&"
 
_user_specified_name231364:&
"
 
_user_specified_name231362:&	"
 
_user_specified_name231360:&"
 
_user_specified_name231358:&"
 
_user_specified_name231356:&"
 
_user_specified_name231354:&"
 
_user_specified_name231352:&"
 
_user_specified_name231350:&"
 
_user_specified_name231348:&"
 
_user_specified_name231346:WS
'
_output_shapes
:���������
(
_user_specified_nameTimeLimitInput:d `
+
_output_shapes
:���������P	
1
_user_specified_nameStackLevelInputFeatures
��
�
Q__inference_transformer_encoder_5_layer_call_and_return_conditional_losses_231151

inputsJ
<encoder_1st_normalizationlayer_3_mul_readvariableop_resource:	J
<encoder_1st_normalizationlayer_3_add_readvariableop_resource:	^
Hencoder_selfattentionlayer_3_query_einsum_einsum_readvariableop_resource:	P
>encoder_selfattentionlayer_3_query_add_readvariableop_resource:\
Fencoder_selfattentionlayer_3_key_einsum_einsum_readvariableop_resource:	N
<encoder_selfattentionlayer_3_key_add_readvariableop_resource:^
Hencoder_selfattentionlayer_3_value_einsum_einsum_readvariableop_resource:	P
>encoder_selfattentionlayer_3_value_add_readvariableop_resource:i
Sencoder_selfattentionlayer_3_attention_output_einsum_einsum_readvariableop_resource:	W
Iencoder_selfattentionlayer_3_attention_output_add_readvariableop_resource:	J
<encoder_2nd_normalizationlayer_3_mul_readvariableop_resource:	J
<encoder_2nd_normalizationlayer_3_add_readvariableop_resource:	P
>encoder_feedforwardlayer_1_3_tensordot_readvariableop_resource:		J
<encoder_feedforwardlayer_1_3_biasadd_readvariableop_resource:	P
>encoder_feedforwardlayer_2_3_tensordot_readvariableop_resource:		J
<encoder_feedforwardlayer_2_3_biasadd_readvariableop_resource:	
identity

identity_1��3Encoder-1st-NormalizationLayer-3/add/ReadVariableOp�3Encoder-1st-NormalizationLayer-3/mul/ReadVariableOp�3Encoder-2nd-NormalizationLayer-3/add/ReadVariableOp�3Encoder-2nd-NormalizationLayer-3/mul/ReadVariableOp�3Encoder-FeedForwardLayer_1_3/BiasAdd/ReadVariableOp�5Encoder-FeedForwardLayer_1_3/Tensordot/ReadVariableOp�3Encoder-FeedForwardLayer_2_3/BiasAdd/ReadVariableOp�5Encoder-FeedForwardLayer_2_3/Tensordot/ReadVariableOp�@Encoder-SelfAttentionLayer-3/attention_output/add/ReadVariableOp�JEncoder-SelfAttentionLayer-3/attention_output/einsum/Einsum/ReadVariableOp�3Encoder-SelfAttentionLayer-3/key/add/ReadVariableOp�=Encoder-SelfAttentionLayer-3/key/einsum/Einsum/ReadVariableOp�5Encoder-SelfAttentionLayer-3/query/add/ReadVariableOp�?Encoder-SelfAttentionLayer-3/query/einsum/Einsum/ReadVariableOp�5Encoder-SelfAttentionLayer-3/value/add/ReadVariableOp�?Encoder-SelfAttentionLayer-3/value/einsum/Einsum/ReadVariableOpj
&Encoder-1st-NormalizationLayer-3/ShapeShapeinputs*
T0*
_output_shapes
::��~
4Encoder-1st-NormalizationLayer-3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
6Encoder-1st-NormalizationLayer-3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
6Encoder-1st-NormalizationLayer-3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
.Encoder-1st-NormalizationLayer-3/strided_sliceStridedSlice/Encoder-1st-NormalizationLayer-3/Shape:output:0=Encoder-1st-NormalizationLayer-3/strided_slice/stack:output:0?Encoder-1st-NormalizationLayer-3/strided_slice/stack_1:output:0?Encoder-1st-NormalizationLayer-3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskp
&Encoder-1st-NormalizationLayer-3/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
%Encoder-1st-NormalizationLayer-3/ProdProd7Encoder-1st-NormalizationLayer-3/strided_slice:output:0/Encoder-1st-NormalizationLayer-3/Const:output:0*
T0*
_output_shapes
: �
6Encoder-1st-NormalizationLayer-3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:�
8Encoder-1st-NormalizationLayer-3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: �
8Encoder-1st-NormalizationLayer-3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
0Encoder-1st-NormalizationLayer-3/strided_slice_1StridedSlice/Encoder-1st-NormalizationLayer-3/Shape:output:0?Encoder-1st-NormalizationLayer-3/strided_slice_1/stack:output:0AEncoder-1st-NormalizationLayer-3/strided_slice_1/stack_1:output:0AEncoder-1st-NormalizationLayer-3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_maskr
(Encoder-1st-NormalizationLayer-3/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
'Encoder-1st-NormalizationLayer-3/Prod_1Prod9Encoder-1st-NormalizationLayer-3/strided_slice_1:output:01Encoder-1st-NormalizationLayer-3/Const_1:output:0*
T0*
_output_shapes
: r
0Encoder-1st-NormalizationLayer-3/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :r
0Encoder-1st-NormalizationLayer-3/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :�
.Encoder-1st-NormalizationLayer-3/Reshape/shapePack9Encoder-1st-NormalizationLayer-3/Reshape/shape/0:output:0.Encoder-1st-NormalizationLayer-3/Prod:output:00Encoder-1st-NormalizationLayer-3/Prod_1:output:09Encoder-1st-NormalizationLayer-3/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:�
(Encoder-1st-NormalizationLayer-3/ReshapeReshapeinputs7Encoder-1st-NormalizationLayer-3/Reshape/shape:output:0*
T0*8
_output_shapes&
$:"�������������������
,Encoder-1st-NormalizationLayer-3/ones/packedPack.Encoder-1st-NormalizationLayer-3/Prod:output:0*
N*
T0*
_output_shapes
:p
+Encoder-1st-NormalizationLayer-3/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
%Encoder-1st-NormalizationLayer-3/onesFill5Encoder-1st-NormalizationLayer-3/ones/packed:output:04Encoder-1st-NormalizationLayer-3/ones/Const:output:0*
T0*#
_output_shapes
:����������
-Encoder-1st-NormalizationLayer-3/zeros/packedPack.Encoder-1st-NormalizationLayer-3/Prod:output:0*
N*
T0*
_output_shapes
:q
,Encoder-1st-NormalizationLayer-3/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
&Encoder-1st-NormalizationLayer-3/zerosFill6Encoder-1st-NormalizationLayer-3/zeros/packed:output:05Encoder-1st-NormalizationLayer-3/zeros/Const:output:0*
T0*#
_output_shapes
:���������k
(Encoder-1st-NormalizationLayer-3/Const_2Const*
_output_shapes
: *
dtype0*
valueB k
(Encoder-1st-NormalizationLayer-3/Const_3Const*
_output_shapes
: *
dtype0*
valueB �
1Encoder-1st-NormalizationLayer-3/FusedBatchNormV3FusedBatchNormV31Encoder-1st-NormalizationLayer-3/Reshape:output:0.Encoder-1st-NormalizationLayer-3/ones:output:0/Encoder-1st-NormalizationLayer-3/zeros:output:01Encoder-1st-NormalizationLayer-3/Const_2:output:01Encoder-1st-NormalizationLayer-3/Const_3:output:0*
T0*
U0*x
_output_shapesf
d:"������������������:���������:���������:���������:���������:*
data_formatNCHW*
epsilon%o�:�
*Encoder-1st-NormalizationLayer-3/Reshape_1Reshape5Encoder-1st-NormalizationLayer-3/FusedBatchNormV3:y:0/Encoder-1st-NormalizationLayer-3/Shape:output:0*
T0*+
_output_shapes
:���������P	�
3Encoder-1st-NormalizationLayer-3/mul/ReadVariableOpReadVariableOp<encoder_1st_normalizationlayer_3_mul_readvariableop_resource*
_output_shapes
:	*
dtype0�
$Encoder-1st-NormalizationLayer-3/mulMul3Encoder-1st-NormalizationLayer-3/Reshape_1:output:0;Encoder-1st-NormalizationLayer-3/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������P	�
3Encoder-1st-NormalizationLayer-3/add/ReadVariableOpReadVariableOp<encoder_1st_normalizationlayer_3_add_readvariableop_resource*
_output_shapes
:	*
dtype0�
$Encoder-1st-NormalizationLayer-3/addAddV2(Encoder-1st-NormalizationLayer-3/mul:z:0;Encoder-1st-NormalizationLayer-3/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������P	�
?Encoder-SelfAttentionLayer-3/query/einsum/Einsum/ReadVariableOpReadVariableOpHencoder_selfattentionlayer_3_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:	*
dtype0�
0Encoder-SelfAttentionLayer-3/query/einsum/EinsumEinsum(Encoder-1st-NormalizationLayer-3/add:z:0GEncoder-SelfAttentionLayer-3/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������P*
equationabc,cde->abde�
5Encoder-SelfAttentionLayer-3/query/add/ReadVariableOpReadVariableOp>encoder_selfattentionlayer_3_query_add_readvariableop_resource*
_output_shapes

:*
dtype0�
&Encoder-SelfAttentionLayer-3/query/addAddV29Encoder-SelfAttentionLayer-3/query/einsum/Einsum:output:0=Encoder-SelfAttentionLayer-3/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������P�
=Encoder-SelfAttentionLayer-3/key/einsum/Einsum/ReadVariableOpReadVariableOpFencoder_selfattentionlayer_3_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:	*
dtype0�
.Encoder-SelfAttentionLayer-3/key/einsum/EinsumEinsum(Encoder-1st-NormalizationLayer-3/add:z:0EEncoder-SelfAttentionLayer-3/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������P*
equationabc,cde->abde�
3Encoder-SelfAttentionLayer-3/key/add/ReadVariableOpReadVariableOp<encoder_selfattentionlayer_3_key_add_readvariableop_resource*
_output_shapes

:*
dtype0�
$Encoder-SelfAttentionLayer-3/key/addAddV27Encoder-SelfAttentionLayer-3/key/einsum/Einsum:output:0;Encoder-SelfAttentionLayer-3/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������P�
?Encoder-SelfAttentionLayer-3/value/einsum/Einsum/ReadVariableOpReadVariableOpHencoder_selfattentionlayer_3_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:	*
dtype0�
0Encoder-SelfAttentionLayer-3/value/einsum/EinsumEinsum(Encoder-1st-NormalizationLayer-3/add:z:0GEncoder-SelfAttentionLayer-3/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������P*
equationabc,cde->abde�
5Encoder-SelfAttentionLayer-3/value/add/ReadVariableOpReadVariableOp>encoder_selfattentionlayer_3_value_add_readvariableop_resource*
_output_shapes

:*
dtype0�
&Encoder-SelfAttentionLayer-3/value/addAddV29Encoder-SelfAttentionLayer-3/value/einsum/Einsum:output:0=Encoder-SelfAttentionLayer-3/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������Pg
"Encoder-SelfAttentionLayer-3/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *:�?�
 Encoder-SelfAttentionLayer-3/MulMul*Encoder-SelfAttentionLayer-3/query/add:z:0+Encoder-SelfAttentionLayer-3/Mul/y:output:0*
T0*/
_output_shapes
:���������P�
*Encoder-SelfAttentionLayer-3/einsum/EinsumEinsum(Encoder-SelfAttentionLayer-3/key/add:z:0$Encoder-SelfAttentionLayer-3/Mul:z:0*
N*
T0*/
_output_shapes
:���������PP*
equationaecd,abcd->acbe�
,Encoder-SelfAttentionLayer-3/softmax/SoftmaxSoftmax3Encoder-SelfAttentionLayer-3/einsum/Einsum:output:0*
T0*/
_output_shapes
:���������PP�
-Encoder-SelfAttentionLayer-3/dropout/IdentityIdentity6Encoder-SelfAttentionLayer-3/softmax/Softmax:softmax:0*
T0*/
_output_shapes
:���������PP�
,Encoder-SelfAttentionLayer-3/einsum_1/EinsumEinsum6Encoder-SelfAttentionLayer-3/dropout/Identity:output:0*Encoder-SelfAttentionLayer-3/value/add:z:0*
N*
T0*/
_output_shapes
:���������P*
equationacbe,aecd->abcd�
JEncoder-SelfAttentionLayer-3/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpSencoder_selfattentionlayer_3_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:	*
dtype0�
;Encoder-SelfAttentionLayer-3/attention_output/einsum/EinsumEinsum5Encoder-SelfAttentionLayer-3/einsum_1/Einsum:output:0REncoder-SelfAttentionLayer-3/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:���������P	*
equationabcd,cde->abe�
@Encoder-SelfAttentionLayer-3/attention_output/add/ReadVariableOpReadVariableOpIencoder_selfattentionlayer_3_attention_output_add_readvariableop_resource*
_output_shapes
:	*
dtype0�
1Encoder-SelfAttentionLayer-3/attention_output/addAddV2DEncoder-SelfAttentionLayer-3/attention_output/einsum/Einsum:output:0HEncoder-SelfAttentionLayer-3/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������P	�
Encoder-1st-AdditionLayer-3/addAddV2inputs5Encoder-SelfAttentionLayer-3/attention_output/add:z:0*
T0*+
_output_shapes
:���������P	�
&Encoder-2nd-NormalizationLayer-3/ShapeShape#Encoder-1st-AdditionLayer-3/add:z:0*
T0*
_output_shapes
::��~
4Encoder-2nd-NormalizationLayer-3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
6Encoder-2nd-NormalizationLayer-3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
6Encoder-2nd-NormalizationLayer-3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
.Encoder-2nd-NormalizationLayer-3/strided_sliceStridedSlice/Encoder-2nd-NormalizationLayer-3/Shape:output:0=Encoder-2nd-NormalizationLayer-3/strided_slice/stack:output:0?Encoder-2nd-NormalizationLayer-3/strided_slice/stack_1:output:0?Encoder-2nd-NormalizationLayer-3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskp
&Encoder-2nd-NormalizationLayer-3/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
%Encoder-2nd-NormalizationLayer-3/ProdProd7Encoder-2nd-NormalizationLayer-3/strided_slice:output:0/Encoder-2nd-NormalizationLayer-3/Const:output:0*
T0*
_output_shapes
: �
6Encoder-2nd-NormalizationLayer-3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:�
8Encoder-2nd-NormalizationLayer-3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: �
8Encoder-2nd-NormalizationLayer-3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
0Encoder-2nd-NormalizationLayer-3/strided_slice_1StridedSlice/Encoder-2nd-NormalizationLayer-3/Shape:output:0?Encoder-2nd-NormalizationLayer-3/strided_slice_1/stack:output:0AEncoder-2nd-NormalizationLayer-3/strided_slice_1/stack_1:output:0AEncoder-2nd-NormalizationLayer-3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_maskr
(Encoder-2nd-NormalizationLayer-3/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
'Encoder-2nd-NormalizationLayer-3/Prod_1Prod9Encoder-2nd-NormalizationLayer-3/strided_slice_1:output:01Encoder-2nd-NormalizationLayer-3/Const_1:output:0*
T0*
_output_shapes
: r
0Encoder-2nd-NormalizationLayer-3/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :r
0Encoder-2nd-NormalizationLayer-3/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :�
.Encoder-2nd-NormalizationLayer-3/Reshape/shapePack9Encoder-2nd-NormalizationLayer-3/Reshape/shape/0:output:0.Encoder-2nd-NormalizationLayer-3/Prod:output:00Encoder-2nd-NormalizationLayer-3/Prod_1:output:09Encoder-2nd-NormalizationLayer-3/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:�
(Encoder-2nd-NormalizationLayer-3/ReshapeReshape#Encoder-1st-AdditionLayer-3/add:z:07Encoder-2nd-NormalizationLayer-3/Reshape/shape:output:0*
T0*8
_output_shapes&
$:"�������������������
,Encoder-2nd-NormalizationLayer-3/ones/packedPack.Encoder-2nd-NormalizationLayer-3/Prod:output:0*
N*
T0*
_output_shapes
:p
+Encoder-2nd-NormalizationLayer-3/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
%Encoder-2nd-NormalizationLayer-3/onesFill5Encoder-2nd-NormalizationLayer-3/ones/packed:output:04Encoder-2nd-NormalizationLayer-3/ones/Const:output:0*
T0*#
_output_shapes
:����������
-Encoder-2nd-NormalizationLayer-3/zeros/packedPack.Encoder-2nd-NormalizationLayer-3/Prod:output:0*
N*
T0*
_output_shapes
:q
,Encoder-2nd-NormalizationLayer-3/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
&Encoder-2nd-NormalizationLayer-3/zerosFill6Encoder-2nd-NormalizationLayer-3/zeros/packed:output:05Encoder-2nd-NormalizationLayer-3/zeros/Const:output:0*
T0*#
_output_shapes
:���������k
(Encoder-2nd-NormalizationLayer-3/Const_2Const*
_output_shapes
: *
dtype0*
valueB k
(Encoder-2nd-NormalizationLayer-3/Const_3Const*
_output_shapes
: *
dtype0*
valueB �
1Encoder-2nd-NormalizationLayer-3/FusedBatchNormV3FusedBatchNormV31Encoder-2nd-NormalizationLayer-3/Reshape:output:0.Encoder-2nd-NormalizationLayer-3/ones:output:0/Encoder-2nd-NormalizationLayer-3/zeros:output:01Encoder-2nd-NormalizationLayer-3/Const_2:output:01Encoder-2nd-NormalizationLayer-3/Const_3:output:0*
T0*
U0*x
_output_shapesf
d:"������������������:���������:���������:���������:���������:*
data_formatNCHW*
epsilon%o�:�
*Encoder-2nd-NormalizationLayer-3/Reshape_1Reshape5Encoder-2nd-NormalizationLayer-3/FusedBatchNormV3:y:0/Encoder-2nd-NormalizationLayer-3/Shape:output:0*
T0*+
_output_shapes
:���������P	�
3Encoder-2nd-NormalizationLayer-3/mul/ReadVariableOpReadVariableOp<encoder_2nd_normalizationlayer_3_mul_readvariableop_resource*
_output_shapes
:	*
dtype0�
$Encoder-2nd-NormalizationLayer-3/mulMul3Encoder-2nd-NormalizationLayer-3/Reshape_1:output:0;Encoder-2nd-NormalizationLayer-3/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������P	�
3Encoder-2nd-NormalizationLayer-3/add/ReadVariableOpReadVariableOp<encoder_2nd_normalizationlayer_3_add_readvariableop_resource*
_output_shapes
:	*
dtype0�
$Encoder-2nd-NormalizationLayer-3/addAddV2(Encoder-2nd-NormalizationLayer-3/mul:z:0;Encoder-2nd-NormalizationLayer-3/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������P	�
5Encoder-FeedForwardLayer_1_3/Tensordot/ReadVariableOpReadVariableOp>encoder_feedforwardlayer_1_3_tensordot_readvariableop_resource*
_output_shapes

:		*
dtype0u
+Encoder-FeedForwardLayer_1_3/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:|
+Encoder-FeedForwardLayer_1_3/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       �
,Encoder-FeedForwardLayer_1_3/Tensordot/ShapeShape(Encoder-2nd-NormalizationLayer-3/add:z:0*
T0*
_output_shapes
::��v
4Encoder-FeedForwardLayer_1_3/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
/Encoder-FeedForwardLayer_1_3/Tensordot/GatherV2GatherV25Encoder-FeedForwardLayer_1_3/Tensordot/Shape:output:04Encoder-FeedForwardLayer_1_3/Tensordot/free:output:0=Encoder-FeedForwardLayer_1_3/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:x
6Encoder-FeedForwardLayer_1_3/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
1Encoder-FeedForwardLayer_1_3/Tensordot/GatherV2_1GatherV25Encoder-FeedForwardLayer_1_3/Tensordot/Shape:output:04Encoder-FeedForwardLayer_1_3/Tensordot/axes:output:0?Encoder-FeedForwardLayer_1_3/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:v
,Encoder-FeedForwardLayer_1_3/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
+Encoder-FeedForwardLayer_1_3/Tensordot/ProdProd8Encoder-FeedForwardLayer_1_3/Tensordot/GatherV2:output:05Encoder-FeedForwardLayer_1_3/Tensordot/Const:output:0*
T0*
_output_shapes
: x
.Encoder-FeedForwardLayer_1_3/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
-Encoder-FeedForwardLayer_1_3/Tensordot/Prod_1Prod:Encoder-FeedForwardLayer_1_3/Tensordot/GatherV2_1:output:07Encoder-FeedForwardLayer_1_3/Tensordot/Const_1:output:0*
T0*
_output_shapes
: t
2Encoder-FeedForwardLayer_1_3/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
-Encoder-FeedForwardLayer_1_3/Tensordot/concatConcatV24Encoder-FeedForwardLayer_1_3/Tensordot/free:output:04Encoder-FeedForwardLayer_1_3/Tensordot/axes:output:0;Encoder-FeedForwardLayer_1_3/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
,Encoder-FeedForwardLayer_1_3/Tensordot/stackPack4Encoder-FeedForwardLayer_1_3/Tensordot/Prod:output:06Encoder-FeedForwardLayer_1_3/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
0Encoder-FeedForwardLayer_1_3/Tensordot/transpose	Transpose(Encoder-2nd-NormalizationLayer-3/add:z:06Encoder-FeedForwardLayer_1_3/Tensordot/concat:output:0*
T0*+
_output_shapes
:���������P	�
.Encoder-FeedForwardLayer_1_3/Tensordot/ReshapeReshape4Encoder-FeedForwardLayer_1_3/Tensordot/transpose:y:05Encoder-FeedForwardLayer_1_3/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
-Encoder-FeedForwardLayer_1_3/Tensordot/MatMulMatMul7Encoder-FeedForwardLayer_1_3/Tensordot/Reshape:output:0=Encoder-FeedForwardLayer_1_3/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������	x
.Encoder-FeedForwardLayer_1_3/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:	v
4Encoder-FeedForwardLayer_1_3/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
/Encoder-FeedForwardLayer_1_3/Tensordot/concat_1ConcatV28Encoder-FeedForwardLayer_1_3/Tensordot/GatherV2:output:07Encoder-FeedForwardLayer_1_3/Tensordot/Const_2:output:0=Encoder-FeedForwardLayer_1_3/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
&Encoder-FeedForwardLayer_1_3/TensordotReshape7Encoder-FeedForwardLayer_1_3/Tensordot/MatMul:product:08Encoder-FeedForwardLayer_1_3/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������P	�
3Encoder-FeedForwardLayer_1_3/BiasAdd/ReadVariableOpReadVariableOp<encoder_feedforwardlayer_1_3_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype0�
$Encoder-FeedForwardLayer_1_3/BiasAddBiasAdd/Encoder-FeedForwardLayer_1_3/Tensordot:output:0;Encoder-FeedForwardLayer_1_3/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������P	l
'Encoder-FeedForwardLayer_1_3/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
%Encoder-FeedForwardLayer_1_3/Gelu/mulMul0Encoder-FeedForwardLayer_1_3/Gelu/mul/x:output:0-Encoder-FeedForwardLayer_1_3/BiasAdd:output:0*
T0*+
_output_shapes
:���������P	m
(Encoder-FeedForwardLayer_1_3/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *��?�
)Encoder-FeedForwardLayer_1_3/Gelu/truedivRealDiv-Encoder-FeedForwardLayer_1_3/BiasAdd:output:01Encoder-FeedForwardLayer_1_3/Gelu/Cast/x:output:0*
T0*+
_output_shapes
:���������P	�
%Encoder-FeedForwardLayer_1_3/Gelu/ErfErf-Encoder-FeedForwardLayer_1_3/Gelu/truediv:z:0*
T0*+
_output_shapes
:���������P	l
'Encoder-FeedForwardLayer_1_3/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
%Encoder-FeedForwardLayer_1_3/Gelu/addAddV20Encoder-FeedForwardLayer_1_3/Gelu/add/x:output:0)Encoder-FeedForwardLayer_1_3/Gelu/Erf:y:0*
T0*+
_output_shapes
:���������P	�
'Encoder-FeedForwardLayer_1_3/Gelu/mul_1Mul)Encoder-FeedForwardLayer_1_3/Gelu/mul:z:0)Encoder-FeedForwardLayer_1_3/Gelu/add:z:0*
T0*+
_output_shapes
:���������P	�
5Encoder-FeedForwardLayer_2_3/Tensordot/ReadVariableOpReadVariableOp>encoder_feedforwardlayer_2_3_tensordot_readvariableop_resource*
_output_shapes

:		*
dtype0u
+Encoder-FeedForwardLayer_2_3/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:|
+Encoder-FeedForwardLayer_2_3/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       �
,Encoder-FeedForwardLayer_2_3/Tensordot/ShapeShape+Encoder-FeedForwardLayer_1_3/Gelu/mul_1:z:0*
T0*
_output_shapes
::��v
4Encoder-FeedForwardLayer_2_3/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
/Encoder-FeedForwardLayer_2_3/Tensordot/GatherV2GatherV25Encoder-FeedForwardLayer_2_3/Tensordot/Shape:output:04Encoder-FeedForwardLayer_2_3/Tensordot/free:output:0=Encoder-FeedForwardLayer_2_3/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:x
6Encoder-FeedForwardLayer_2_3/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
1Encoder-FeedForwardLayer_2_3/Tensordot/GatherV2_1GatherV25Encoder-FeedForwardLayer_2_3/Tensordot/Shape:output:04Encoder-FeedForwardLayer_2_3/Tensordot/axes:output:0?Encoder-FeedForwardLayer_2_3/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:v
,Encoder-FeedForwardLayer_2_3/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
+Encoder-FeedForwardLayer_2_3/Tensordot/ProdProd8Encoder-FeedForwardLayer_2_3/Tensordot/GatherV2:output:05Encoder-FeedForwardLayer_2_3/Tensordot/Const:output:0*
T0*
_output_shapes
: x
.Encoder-FeedForwardLayer_2_3/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
-Encoder-FeedForwardLayer_2_3/Tensordot/Prod_1Prod:Encoder-FeedForwardLayer_2_3/Tensordot/GatherV2_1:output:07Encoder-FeedForwardLayer_2_3/Tensordot/Const_1:output:0*
T0*
_output_shapes
: t
2Encoder-FeedForwardLayer_2_3/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
-Encoder-FeedForwardLayer_2_3/Tensordot/concatConcatV24Encoder-FeedForwardLayer_2_3/Tensordot/free:output:04Encoder-FeedForwardLayer_2_3/Tensordot/axes:output:0;Encoder-FeedForwardLayer_2_3/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
,Encoder-FeedForwardLayer_2_3/Tensordot/stackPack4Encoder-FeedForwardLayer_2_3/Tensordot/Prod:output:06Encoder-FeedForwardLayer_2_3/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
0Encoder-FeedForwardLayer_2_3/Tensordot/transpose	Transpose+Encoder-FeedForwardLayer_1_3/Gelu/mul_1:z:06Encoder-FeedForwardLayer_2_3/Tensordot/concat:output:0*
T0*+
_output_shapes
:���������P	�
.Encoder-FeedForwardLayer_2_3/Tensordot/ReshapeReshape4Encoder-FeedForwardLayer_2_3/Tensordot/transpose:y:05Encoder-FeedForwardLayer_2_3/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
-Encoder-FeedForwardLayer_2_3/Tensordot/MatMulMatMul7Encoder-FeedForwardLayer_2_3/Tensordot/Reshape:output:0=Encoder-FeedForwardLayer_2_3/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������	x
.Encoder-FeedForwardLayer_2_3/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:	v
4Encoder-FeedForwardLayer_2_3/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
/Encoder-FeedForwardLayer_2_3/Tensordot/concat_1ConcatV28Encoder-FeedForwardLayer_2_3/Tensordot/GatherV2:output:07Encoder-FeedForwardLayer_2_3/Tensordot/Const_2:output:0=Encoder-FeedForwardLayer_2_3/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
&Encoder-FeedForwardLayer_2_3/TensordotReshape7Encoder-FeedForwardLayer_2_3/Tensordot/MatMul:product:08Encoder-FeedForwardLayer_2_3/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������P	�
3Encoder-FeedForwardLayer_2_3/BiasAdd/ReadVariableOpReadVariableOp<encoder_feedforwardlayer_2_3_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype0�
$Encoder-FeedForwardLayer_2_3/BiasAddBiasAdd/Encoder-FeedForwardLayer_2_3/Tensordot:output:0;Encoder-FeedForwardLayer_2_3/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������P	�
dropout_5/IdentityIdentity-Encoder-FeedForwardLayer_2_3/BiasAdd:output:0*
T0*+
_output_shapes
:���������P	�
Encoder-2nd-AdditionLayer-3/addAddV2#Encoder-1st-AdditionLayer-3/add:z:0dropout_5/Identity:output:0*
T0*+
_output_shapes
:���������P	v
IdentityIdentity#Encoder-2nd-AdditionLayer-3/add:z:0^NoOp*
T0*+
_output_shapes
:���������P	�

Identity_1Identity6Encoder-SelfAttentionLayer-3/softmax/Softmax:softmax:0^NoOp*
T0*/
_output_shapes
:���������PP�
NoOpNoOp4^Encoder-1st-NormalizationLayer-3/add/ReadVariableOp4^Encoder-1st-NormalizationLayer-3/mul/ReadVariableOp4^Encoder-2nd-NormalizationLayer-3/add/ReadVariableOp4^Encoder-2nd-NormalizationLayer-3/mul/ReadVariableOp4^Encoder-FeedForwardLayer_1_3/BiasAdd/ReadVariableOp6^Encoder-FeedForwardLayer_1_3/Tensordot/ReadVariableOp4^Encoder-FeedForwardLayer_2_3/BiasAdd/ReadVariableOp6^Encoder-FeedForwardLayer_2_3/Tensordot/ReadVariableOpA^Encoder-SelfAttentionLayer-3/attention_output/add/ReadVariableOpK^Encoder-SelfAttentionLayer-3/attention_output/einsum/Einsum/ReadVariableOp4^Encoder-SelfAttentionLayer-3/key/add/ReadVariableOp>^Encoder-SelfAttentionLayer-3/key/einsum/Einsum/ReadVariableOp6^Encoder-SelfAttentionLayer-3/query/add/ReadVariableOp@^Encoder-SelfAttentionLayer-3/query/einsum/Einsum/ReadVariableOp6^Encoder-SelfAttentionLayer-3/value/add/ReadVariableOp@^Encoder-SelfAttentionLayer-3/value/einsum/Einsum/ReadVariableOp*
_output_shapes
 "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:���������P	: : : : : : : : : : : : : : : : 2j
3Encoder-1st-NormalizationLayer-3/add/ReadVariableOp3Encoder-1st-NormalizationLayer-3/add/ReadVariableOp2j
3Encoder-1st-NormalizationLayer-3/mul/ReadVariableOp3Encoder-1st-NormalizationLayer-3/mul/ReadVariableOp2j
3Encoder-2nd-NormalizationLayer-3/add/ReadVariableOp3Encoder-2nd-NormalizationLayer-3/add/ReadVariableOp2j
3Encoder-2nd-NormalizationLayer-3/mul/ReadVariableOp3Encoder-2nd-NormalizationLayer-3/mul/ReadVariableOp2j
3Encoder-FeedForwardLayer_1_3/BiasAdd/ReadVariableOp3Encoder-FeedForwardLayer_1_3/BiasAdd/ReadVariableOp2n
5Encoder-FeedForwardLayer_1_3/Tensordot/ReadVariableOp5Encoder-FeedForwardLayer_1_3/Tensordot/ReadVariableOp2j
3Encoder-FeedForwardLayer_2_3/BiasAdd/ReadVariableOp3Encoder-FeedForwardLayer_2_3/BiasAdd/ReadVariableOp2n
5Encoder-FeedForwardLayer_2_3/Tensordot/ReadVariableOp5Encoder-FeedForwardLayer_2_3/Tensordot/ReadVariableOp2�
@Encoder-SelfAttentionLayer-3/attention_output/add/ReadVariableOp@Encoder-SelfAttentionLayer-3/attention_output/add/ReadVariableOp2�
JEncoder-SelfAttentionLayer-3/attention_output/einsum/Einsum/ReadVariableOpJEncoder-SelfAttentionLayer-3/attention_output/einsum/Einsum/ReadVariableOp2j
3Encoder-SelfAttentionLayer-3/key/add/ReadVariableOp3Encoder-SelfAttentionLayer-3/key/add/ReadVariableOp2~
=Encoder-SelfAttentionLayer-3/key/einsum/Einsum/ReadVariableOp=Encoder-SelfAttentionLayer-3/key/einsum/Einsum/ReadVariableOp2n
5Encoder-SelfAttentionLayer-3/query/add/ReadVariableOp5Encoder-SelfAttentionLayer-3/query/add/ReadVariableOp2�
?Encoder-SelfAttentionLayer-3/query/einsum/Einsum/ReadVariableOp?Encoder-SelfAttentionLayer-3/query/einsum/Einsum/ReadVariableOp2n
5Encoder-SelfAttentionLayer-3/value/add/ReadVariableOp5Encoder-SelfAttentionLayer-3/value/add/ReadVariableOp2�
?Encoder-SelfAttentionLayer-3/value/einsum/Einsum/ReadVariableOp?Encoder-SelfAttentionLayer-3/value/einsum/Einsum/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:(
$
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:S O
+
_output_shapes
:���������P	
 
_user_specified_nameinputs
��
�B
__inference__traced_save_233739
file_prefix9
+read_disablecopyonread_finallayernorm_gamma:	:
,read_1_disablecopyonread_finallayernorm_beta:	P
>read_2_disablecopyonread_fullyconnectedlayerimprovement_kernel:

J
<read_3_disablecopyonread_fullyconnectedlayerimprovement_bias:
G
5read_4_disablecopyonread_predictionimprovement_kernel:
A
3read_5_disablecopyonread_predictionimprovement_bias:n
Xread_6_disablecopyonread_transformer_encoder_3_encoder_selfattentionlayer_1_query_kernel:	h
Vread_7_disablecopyonread_transformer_encoder_3_encoder_selfattentionlayer_1_query_bias:l
Vread_8_disablecopyonread_transformer_encoder_3_encoder_selfattentionlayer_1_key_kernel:	f
Tread_9_disablecopyonread_transformer_encoder_3_encoder_selfattentionlayer_1_key_bias:o
Yread_10_disablecopyonread_transformer_encoder_3_encoder_selfattentionlayer_1_value_kernel:	i
Wread_11_disablecopyonread_transformer_encoder_3_encoder_selfattentionlayer_1_value_bias:z
dread_12_disablecopyonread_transformer_encoder_3_encoder_selfattentionlayer_1_attention_output_kernel:	p
bread_13_disablecopyonread_transformer_encoder_3_encoder_selfattentionlayer_1_attention_output_bias:	d
Vread_14_disablecopyonread_transformer_encoder_3_encoder_1st_normalizationlayer_1_gamma:	c
Uread_15_disablecopyonread_transformer_encoder_3_encoder_1st_normalizationlayer_1_beta:	d
Vread_16_disablecopyonread_transformer_encoder_3_encoder_2nd_normalizationlayer_1_gamma:	c
Uread_17_disablecopyonread_transformer_encoder_3_encoder_2nd_normalizationlayer_1_beta:	e
Sread_18_disablecopyonread_transformer_encoder_3_encoder_feedforwardlayer_1_1_kernel:		_
Qread_19_disablecopyonread_transformer_encoder_3_encoder_feedforwardlayer_1_1_bias:	e
Sread_20_disablecopyonread_transformer_encoder_3_encoder_feedforwardlayer_2_1_kernel:		_
Qread_21_disablecopyonread_transformer_encoder_3_encoder_feedforwardlayer_2_1_bias:	o
Yread_22_disablecopyonread_transformer_encoder_4_encoder_selfattentionlayer_2_query_kernel:	i
Wread_23_disablecopyonread_transformer_encoder_4_encoder_selfattentionlayer_2_query_bias:m
Wread_24_disablecopyonread_transformer_encoder_4_encoder_selfattentionlayer_2_key_kernel:	g
Uread_25_disablecopyonread_transformer_encoder_4_encoder_selfattentionlayer_2_key_bias:o
Yread_26_disablecopyonread_transformer_encoder_4_encoder_selfattentionlayer_2_value_kernel:	i
Wread_27_disablecopyonread_transformer_encoder_4_encoder_selfattentionlayer_2_value_bias:z
dread_28_disablecopyonread_transformer_encoder_4_encoder_selfattentionlayer_2_attention_output_kernel:	p
bread_29_disablecopyonread_transformer_encoder_4_encoder_selfattentionlayer_2_attention_output_bias:	d
Vread_30_disablecopyonread_transformer_encoder_4_encoder_1st_normalizationlayer_2_gamma:	c
Uread_31_disablecopyonread_transformer_encoder_4_encoder_1st_normalizationlayer_2_beta:	d
Vread_32_disablecopyonread_transformer_encoder_4_encoder_2nd_normalizationlayer_2_gamma:	c
Uread_33_disablecopyonread_transformer_encoder_4_encoder_2nd_normalizationlayer_2_beta:	e
Sread_34_disablecopyonread_transformer_encoder_4_encoder_feedforwardlayer_1_2_kernel:		_
Qread_35_disablecopyonread_transformer_encoder_4_encoder_feedforwardlayer_1_2_bias:	e
Sread_36_disablecopyonread_transformer_encoder_4_encoder_feedforwardlayer_2_2_kernel:		_
Qread_37_disablecopyonread_transformer_encoder_4_encoder_feedforwardlayer_2_2_bias:	o
Yread_38_disablecopyonread_transformer_encoder_5_encoder_selfattentionlayer_3_query_kernel:	i
Wread_39_disablecopyonread_transformer_encoder_5_encoder_selfattentionlayer_3_query_bias:m
Wread_40_disablecopyonread_transformer_encoder_5_encoder_selfattentionlayer_3_key_kernel:	g
Uread_41_disablecopyonread_transformer_encoder_5_encoder_selfattentionlayer_3_key_bias:o
Yread_42_disablecopyonread_transformer_encoder_5_encoder_selfattentionlayer_3_value_kernel:	i
Wread_43_disablecopyonread_transformer_encoder_5_encoder_selfattentionlayer_3_value_bias:z
dread_44_disablecopyonread_transformer_encoder_5_encoder_selfattentionlayer_3_attention_output_kernel:	p
bread_45_disablecopyonread_transformer_encoder_5_encoder_selfattentionlayer_3_attention_output_bias:	d
Vread_46_disablecopyonread_transformer_encoder_5_encoder_1st_normalizationlayer_3_gamma:	c
Uread_47_disablecopyonread_transformer_encoder_5_encoder_1st_normalizationlayer_3_beta:	d
Vread_48_disablecopyonread_transformer_encoder_5_encoder_2nd_normalizationlayer_3_gamma:	c
Uread_49_disablecopyonread_transformer_encoder_5_encoder_2nd_normalizationlayer_3_beta:	e
Sread_50_disablecopyonread_transformer_encoder_5_encoder_feedforwardlayer_1_3_kernel:		_
Qread_51_disablecopyonread_transformer_encoder_5_encoder_feedforwardlayer_1_3_bias:	e
Sread_52_disablecopyonread_transformer_encoder_5_encoder_feedforwardlayer_2_3_kernel:		_
Qread_53_disablecopyonread_transformer_encoder_5_encoder_feedforwardlayer_2_3_bias:	
savev2_const
identity_109��MergeV2Checkpoints�Read/DisableCopyOnRead�Read/ReadVariableOp�Read_1/DisableCopyOnRead�Read_1/ReadVariableOp�Read_10/DisableCopyOnRead�Read_10/ReadVariableOp�Read_11/DisableCopyOnRead�Read_11/ReadVariableOp�Read_12/DisableCopyOnRead�Read_12/ReadVariableOp�Read_13/DisableCopyOnRead�Read_13/ReadVariableOp�Read_14/DisableCopyOnRead�Read_14/ReadVariableOp�Read_15/DisableCopyOnRead�Read_15/ReadVariableOp�Read_16/DisableCopyOnRead�Read_16/ReadVariableOp�Read_17/DisableCopyOnRead�Read_17/ReadVariableOp�Read_18/DisableCopyOnRead�Read_18/ReadVariableOp�Read_19/DisableCopyOnRead�Read_19/ReadVariableOp�Read_2/DisableCopyOnRead�Read_2/ReadVariableOp�Read_20/DisableCopyOnRead�Read_20/ReadVariableOp�Read_21/DisableCopyOnRead�Read_21/ReadVariableOp�Read_22/DisableCopyOnRead�Read_22/ReadVariableOp�Read_23/DisableCopyOnRead�Read_23/ReadVariableOp�Read_24/DisableCopyOnRead�Read_24/ReadVariableOp�Read_25/DisableCopyOnRead�Read_25/ReadVariableOp�Read_26/DisableCopyOnRead�Read_26/ReadVariableOp�Read_27/DisableCopyOnRead�Read_27/ReadVariableOp�Read_28/DisableCopyOnRead�Read_28/ReadVariableOp�Read_29/DisableCopyOnRead�Read_29/ReadVariableOp�Read_3/DisableCopyOnRead�Read_3/ReadVariableOp�Read_30/DisableCopyOnRead�Read_30/ReadVariableOp�Read_31/DisableCopyOnRead�Read_31/ReadVariableOp�Read_32/DisableCopyOnRead�Read_32/ReadVariableOp�Read_33/DisableCopyOnRead�Read_33/ReadVariableOp�Read_34/DisableCopyOnRead�Read_34/ReadVariableOp�Read_35/DisableCopyOnRead�Read_35/ReadVariableOp�Read_36/DisableCopyOnRead�Read_36/ReadVariableOp�Read_37/DisableCopyOnRead�Read_37/ReadVariableOp�Read_38/DisableCopyOnRead�Read_38/ReadVariableOp�Read_39/DisableCopyOnRead�Read_39/ReadVariableOp�Read_4/DisableCopyOnRead�Read_4/ReadVariableOp�Read_40/DisableCopyOnRead�Read_40/ReadVariableOp�Read_41/DisableCopyOnRead�Read_41/ReadVariableOp�Read_42/DisableCopyOnRead�Read_42/ReadVariableOp�Read_43/DisableCopyOnRead�Read_43/ReadVariableOp�Read_44/DisableCopyOnRead�Read_44/ReadVariableOp�Read_45/DisableCopyOnRead�Read_45/ReadVariableOp�Read_46/DisableCopyOnRead�Read_46/ReadVariableOp�Read_47/DisableCopyOnRead�Read_47/ReadVariableOp�Read_48/DisableCopyOnRead�Read_48/ReadVariableOp�Read_49/DisableCopyOnRead�Read_49/ReadVariableOp�Read_5/DisableCopyOnRead�Read_5/ReadVariableOp�Read_50/DisableCopyOnRead�Read_50/ReadVariableOp�Read_51/DisableCopyOnRead�Read_51/ReadVariableOp�Read_52/DisableCopyOnRead�Read_52/ReadVariableOp�Read_53/DisableCopyOnRead�Read_53/ReadVariableOp�Read_6/DisableCopyOnRead�Read_6/ReadVariableOp�Read_7/DisableCopyOnRead�Read_7/ReadVariableOp�Read_8/DisableCopyOnRead�Read_8/ReadVariableOp�Read_9/DisableCopyOnRead�Read_9/ReadVariableOpw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: }
Read/DisableCopyOnReadDisableCopyOnRead+read_disablecopyonread_finallayernorm_gamma"/device:CPU:0*
_output_shapes
 �
Read/ReadVariableOpReadVariableOp+read_disablecopyonread_finallayernorm_gamma^Read/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	*
dtype0e
IdentityIdentityRead/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	]

Identity_1IdentityIdentity:output:0"/device:CPU:0*
T0*
_output_shapes
:	�
Read_1/DisableCopyOnReadDisableCopyOnRead,read_1_disablecopyonread_finallayernorm_beta"/device:CPU:0*
_output_shapes
 �
Read_1/ReadVariableOpReadVariableOp,read_1_disablecopyonread_finallayernorm_beta^Read_1/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	*
dtype0i

Identity_2IdentityRead_1/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	_

Identity_3IdentityIdentity_2:output:0"/device:CPU:0*
T0*
_output_shapes
:	�
Read_2/DisableCopyOnReadDisableCopyOnRead>read_2_disablecopyonread_fullyconnectedlayerimprovement_kernel"/device:CPU:0*
_output_shapes
 �
Read_2/ReadVariableOpReadVariableOp>read_2_disablecopyonread_fullyconnectedlayerimprovement_kernel^Read_2/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:

*
dtype0m

Identity_4IdentityRead_2/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:

c

Identity_5IdentityIdentity_4:output:0"/device:CPU:0*
T0*
_output_shapes

:

�
Read_3/DisableCopyOnReadDisableCopyOnRead<read_3_disablecopyonread_fullyconnectedlayerimprovement_bias"/device:CPU:0*
_output_shapes
 �
Read_3/ReadVariableOpReadVariableOp<read_3_disablecopyonread_fullyconnectedlayerimprovement_bias^Read_3/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:
*
dtype0i

Identity_6IdentityRead_3/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:
_

Identity_7IdentityIdentity_6:output:0"/device:CPU:0*
T0*
_output_shapes
:
�
Read_4/DisableCopyOnReadDisableCopyOnRead5read_4_disablecopyonread_predictionimprovement_kernel"/device:CPU:0*
_output_shapes
 �
Read_4/ReadVariableOpReadVariableOp5read_4_disablecopyonread_predictionimprovement_kernel^Read_4/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:
*
dtype0m

Identity_8IdentityRead_4/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:
c

Identity_9IdentityIdentity_8:output:0"/device:CPU:0*
T0*
_output_shapes

:
�
Read_5/DisableCopyOnReadDisableCopyOnRead3read_5_disablecopyonread_predictionimprovement_bias"/device:CPU:0*
_output_shapes
 �
Read_5/ReadVariableOpReadVariableOp3read_5_disablecopyonread_predictionimprovement_bias^Read_5/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0j
Identity_10IdentityRead_5/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_11IdentityIdentity_10:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_6/DisableCopyOnReadDisableCopyOnReadXread_6_disablecopyonread_transformer_encoder_3_encoder_selfattentionlayer_1_query_kernel"/device:CPU:0*
_output_shapes
 �
Read_6/ReadVariableOpReadVariableOpXread_6_disablecopyonread_transformer_encoder_3_encoder_selfattentionlayer_1_query_kernel^Read_6/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
:	*
dtype0r
Identity_12IdentityRead_6/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
:	i
Identity_13IdentityIdentity_12:output:0"/device:CPU:0*
T0*"
_output_shapes
:	�
Read_7/DisableCopyOnReadDisableCopyOnReadVread_7_disablecopyonread_transformer_encoder_3_encoder_selfattentionlayer_1_query_bias"/device:CPU:0*
_output_shapes
 �
Read_7/ReadVariableOpReadVariableOpVread_7_disablecopyonread_transformer_encoder_3_encoder_selfattentionlayer_1_query_bias^Read_7/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0n
Identity_14IdentityRead_7/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:e
Identity_15IdentityIdentity_14:output:0"/device:CPU:0*
T0*
_output_shapes

:�
Read_8/DisableCopyOnReadDisableCopyOnReadVread_8_disablecopyonread_transformer_encoder_3_encoder_selfattentionlayer_1_key_kernel"/device:CPU:0*
_output_shapes
 �
Read_8/ReadVariableOpReadVariableOpVread_8_disablecopyonread_transformer_encoder_3_encoder_selfattentionlayer_1_key_kernel^Read_8/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
:	*
dtype0r
Identity_16IdentityRead_8/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
:	i
Identity_17IdentityIdentity_16:output:0"/device:CPU:0*
T0*"
_output_shapes
:	�
Read_9/DisableCopyOnReadDisableCopyOnReadTread_9_disablecopyonread_transformer_encoder_3_encoder_selfattentionlayer_1_key_bias"/device:CPU:0*
_output_shapes
 �
Read_9/ReadVariableOpReadVariableOpTread_9_disablecopyonread_transformer_encoder_3_encoder_selfattentionlayer_1_key_bias^Read_9/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0n
Identity_18IdentityRead_9/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:e
Identity_19IdentityIdentity_18:output:0"/device:CPU:0*
T0*
_output_shapes

:�
Read_10/DisableCopyOnReadDisableCopyOnReadYread_10_disablecopyonread_transformer_encoder_3_encoder_selfattentionlayer_1_value_kernel"/device:CPU:0*
_output_shapes
 �
Read_10/ReadVariableOpReadVariableOpYread_10_disablecopyonread_transformer_encoder_3_encoder_selfattentionlayer_1_value_kernel^Read_10/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
:	*
dtype0s
Identity_20IdentityRead_10/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
:	i
Identity_21IdentityIdentity_20:output:0"/device:CPU:0*
T0*"
_output_shapes
:	�
Read_11/DisableCopyOnReadDisableCopyOnReadWread_11_disablecopyonread_transformer_encoder_3_encoder_selfattentionlayer_1_value_bias"/device:CPU:0*
_output_shapes
 �
Read_11/ReadVariableOpReadVariableOpWread_11_disablecopyonread_transformer_encoder_3_encoder_selfattentionlayer_1_value_bias^Read_11/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0o
Identity_22IdentityRead_11/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:e
Identity_23IdentityIdentity_22:output:0"/device:CPU:0*
T0*
_output_shapes

:�
Read_12/DisableCopyOnReadDisableCopyOnReaddread_12_disablecopyonread_transformer_encoder_3_encoder_selfattentionlayer_1_attention_output_kernel"/device:CPU:0*
_output_shapes
 �
Read_12/ReadVariableOpReadVariableOpdread_12_disablecopyonread_transformer_encoder_3_encoder_selfattentionlayer_1_attention_output_kernel^Read_12/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
:	*
dtype0s
Identity_24IdentityRead_12/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
:	i
Identity_25IdentityIdentity_24:output:0"/device:CPU:0*
T0*"
_output_shapes
:	�
Read_13/DisableCopyOnReadDisableCopyOnReadbread_13_disablecopyonread_transformer_encoder_3_encoder_selfattentionlayer_1_attention_output_bias"/device:CPU:0*
_output_shapes
 �
Read_13/ReadVariableOpReadVariableOpbread_13_disablecopyonread_transformer_encoder_3_encoder_selfattentionlayer_1_attention_output_bias^Read_13/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	*
dtype0k
Identity_26IdentityRead_13/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	a
Identity_27IdentityIdentity_26:output:0"/device:CPU:0*
T0*
_output_shapes
:	�
Read_14/DisableCopyOnReadDisableCopyOnReadVread_14_disablecopyonread_transformer_encoder_3_encoder_1st_normalizationlayer_1_gamma"/device:CPU:0*
_output_shapes
 �
Read_14/ReadVariableOpReadVariableOpVread_14_disablecopyonread_transformer_encoder_3_encoder_1st_normalizationlayer_1_gamma^Read_14/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	*
dtype0k
Identity_28IdentityRead_14/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	a
Identity_29IdentityIdentity_28:output:0"/device:CPU:0*
T0*
_output_shapes
:	�
Read_15/DisableCopyOnReadDisableCopyOnReadUread_15_disablecopyonread_transformer_encoder_3_encoder_1st_normalizationlayer_1_beta"/device:CPU:0*
_output_shapes
 �
Read_15/ReadVariableOpReadVariableOpUread_15_disablecopyonread_transformer_encoder_3_encoder_1st_normalizationlayer_1_beta^Read_15/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	*
dtype0k
Identity_30IdentityRead_15/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	a
Identity_31IdentityIdentity_30:output:0"/device:CPU:0*
T0*
_output_shapes
:	�
Read_16/DisableCopyOnReadDisableCopyOnReadVread_16_disablecopyonread_transformer_encoder_3_encoder_2nd_normalizationlayer_1_gamma"/device:CPU:0*
_output_shapes
 �
Read_16/ReadVariableOpReadVariableOpVread_16_disablecopyonread_transformer_encoder_3_encoder_2nd_normalizationlayer_1_gamma^Read_16/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	*
dtype0k
Identity_32IdentityRead_16/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	a
Identity_33IdentityIdentity_32:output:0"/device:CPU:0*
T0*
_output_shapes
:	�
Read_17/DisableCopyOnReadDisableCopyOnReadUread_17_disablecopyonread_transformer_encoder_3_encoder_2nd_normalizationlayer_1_beta"/device:CPU:0*
_output_shapes
 �
Read_17/ReadVariableOpReadVariableOpUread_17_disablecopyonread_transformer_encoder_3_encoder_2nd_normalizationlayer_1_beta^Read_17/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	*
dtype0k
Identity_34IdentityRead_17/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	a
Identity_35IdentityIdentity_34:output:0"/device:CPU:0*
T0*
_output_shapes
:	�
Read_18/DisableCopyOnReadDisableCopyOnReadSread_18_disablecopyonread_transformer_encoder_3_encoder_feedforwardlayer_1_1_kernel"/device:CPU:0*
_output_shapes
 �
Read_18/ReadVariableOpReadVariableOpSread_18_disablecopyonread_transformer_encoder_3_encoder_feedforwardlayer_1_1_kernel^Read_18/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:		*
dtype0o
Identity_36IdentityRead_18/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:		e
Identity_37IdentityIdentity_36:output:0"/device:CPU:0*
T0*
_output_shapes

:		�
Read_19/DisableCopyOnReadDisableCopyOnReadQread_19_disablecopyonread_transformer_encoder_3_encoder_feedforwardlayer_1_1_bias"/device:CPU:0*
_output_shapes
 �
Read_19/ReadVariableOpReadVariableOpQread_19_disablecopyonread_transformer_encoder_3_encoder_feedforwardlayer_1_1_bias^Read_19/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	*
dtype0k
Identity_38IdentityRead_19/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	a
Identity_39IdentityIdentity_38:output:0"/device:CPU:0*
T0*
_output_shapes
:	�
Read_20/DisableCopyOnReadDisableCopyOnReadSread_20_disablecopyonread_transformer_encoder_3_encoder_feedforwardlayer_2_1_kernel"/device:CPU:0*
_output_shapes
 �
Read_20/ReadVariableOpReadVariableOpSread_20_disablecopyonread_transformer_encoder_3_encoder_feedforwardlayer_2_1_kernel^Read_20/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:		*
dtype0o
Identity_40IdentityRead_20/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:		e
Identity_41IdentityIdentity_40:output:0"/device:CPU:0*
T0*
_output_shapes

:		�
Read_21/DisableCopyOnReadDisableCopyOnReadQread_21_disablecopyonread_transformer_encoder_3_encoder_feedforwardlayer_2_1_bias"/device:CPU:0*
_output_shapes
 �
Read_21/ReadVariableOpReadVariableOpQread_21_disablecopyonread_transformer_encoder_3_encoder_feedforwardlayer_2_1_bias^Read_21/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	*
dtype0k
Identity_42IdentityRead_21/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	a
Identity_43IdentityIdentity_42:output:0"/device:CPU:0*
T0*
_output_shapes
:	�
Read_22/DisableCopyOnReadDisableCopyOnReadYread_22_disablecopyonread_transformer_encoder_4_encoder_selfattentionlayer_2_query_kernel"/device:CPU:0*
_output_shapes
 �
Read_22/ReadVariableOpReadVariableOpYread_22_disablecopyonread_transformer_encoder_4_encoder_selfattentionlayer_2_query_kernel^Read_22/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
:	*
dtype0s
Identity_44IdentityRead_22/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
:	i
Identity_45IdentityIdentity_44:output:0"/device:CPU:0*
T0*"
_output_shapes
:	�
Read_23/DisableCopyOnReadDisableCopyOnReadWread_23_disablecopyonread_transformer_encoder_4_encoder_selfattentionlayer_2_query_bias"/device:CPU:0*
_output_shapes
 �
Read_23/ReadVariableOpReadVariableOpWread_23_disablecopyonread_transformer_encoder_4_encoder_selfattentionlayer_2_query_bias^Read_23/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0o
Identity_46IdentityRead_23/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:e
Identity_47IdentityIdentity_46:output:0"/device:CPU:0*
T0*
_output_shapes

:�
Read_24/DisableCopyOnReadDisableCopyOnReadWread_24_disablecopyonread_transformer_encoder_4_encoder_selfattentionlayer_2_key_kernel"/device:CPU:0*
_output_shapes
 �
Read_24/ReadVariableOpReadVariableOpWread_24_disablecopyonread_transformer_encoder_4_encoder_selfattentionlayer_2_key_kernel^Read_24/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
:	*
dtype0s
Identity_48IdentityRead_24/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
:	i
Identity_49IdentityIdentity_48:output:0"/device:CPU:0*
T0*"
_output_shapes
:	�
Read_25/DisableCopyOnReadDisableCopyOnReadUread_25_disablecopyonread_transformer_encoder_4_encoder_selfattentionlayer_2_key_bias"/device:CPU:0*
_output_shapes
 �
Read_25/ReadVariableOpReadVariableOpUread_25_disablecopyonread_transformer_encoder_4_encoder_selfattentionlayer_2_key_bias^Read_25/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0o
Identity_50IdentityRead_25/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:e
Identity_51IdentityIdentity_50:output:0"/device:CPU:0*
T0*
_output_shapes

:�
Read_26/DisableCopyOnReadDisableCopyOnReadYread_26_disablecopyonread_transformer_encoder_4_encoder_selfattentionlayer_2_value_kernel"/device:CPU:0*
_output_shapes
 �
Read_26/ReadVariableOpReadVariableOpYread_26_disablecopyonread_transformer_encoder_4_encoder_selfattentionlayer_2_value_kernel^Read_26/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
:	*
dtype0s
Identity_52IdentityRead_26/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
:	i
Identity_53IdentityIdentity_52:output:0"/device:CPU:0*
T0*"
_output_shapes
:	�
Read_27/DisableCopyOnReadDisableCopyOnReadWread_27_disablecopyonread_transformer_encoder_4_encoder_selfattentionlayer_2_value_bias"/device:CPU:0*
_output_shapes
 �
Read_27/ReadVariableOpReadVariableOpWread_27_disablecopyonread_transformer_encoder_4_encoder_selfattentionlayer_2_value_bias^Read_27/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0o
Identity_54IdentityRead_27/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:e
Identity_55IdentityIdentity_54:output:0"/device:CPU:0*
T0*
_output_shapes

:�
Read_28/DisableCopyOnReadDisableCopyOnReaddread_28_disablecopyonread_transformer_encoder_4_encoder_selfattentionlayer_2_attention_output_kernel"/device:CPU:0*
_output_shapes
 �
Read_28/ReadVariableOpReadVariableOpdread_28_disablecopyonread_transformer_encoder_4_encoder_selfattentionlayer_2_attention_output_kernel^Read_28/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
:	*
dtype0s
Identity_56IdentityRead_28/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
:	i
Identity_57IdentityIdentity_56:output:0"/device:CPU:0*
T0*"
_output_shapes
:	�
Read_29/DisableCopyOnReadDisableCopyOnReadbread_29_disablecopyonread_transformer_encoder_4_encoder_selfattentionlayer_2_attention_output_bias"/device:CPU:0*
_output_shapes
 �
Read_29/ReadVariableOpReadVariableOpbread_29_disablecopyonread_transformer_encoder_4_encoder_selfattentionlayer_2_attention_output_bias^Read_29/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	*
dtype0k
Identity_58IdentityRead_29/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	a
Identity_59IdentityIdentity_58:output:0"/device:CPU:0*
T0*
_output_shapes
:	�
Read_30/DisableCopyOnReadDisableCopyOnReadVread_30_disablecopyonread_transformer_encoder_4_encoder_1st_normalizationlayer_2_gamma"/device:CPU:0*
_output_shapes
 �
Read_30/ReadVariableOpReadVariableOpVread_30_disablecopyonread_transformer_encoder_4_encoder_1st_normalizationlayer_2_gamma^Read_30/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	*
dtype0k
Identity_60IdentityRead_30/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	a
Identity_61IdentityIdentity_60:output:0"/device:CPU:0*
T0*
_output_shapes
:	�
Read_31/DisableCopyOnReadDisableCopyOnReadUread_31_disablecopyonread_transformer_encoder_4_encoder_1st_normalizationlayer_2_beta"/device:CPU:0*
_output_shapes
 �
Read_31/ReadVariableOpReadVariableOpUread_31_disablecopyonread_transformer_encoder_4_encoder_1st_normalizationlayer_2_beta^Read_31/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	*
dtype0k
Identity_62IdentityRead_31/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	a
Identity_63IdentityIdentity_62:output:0"/device:CPU:0*
T0*
_output_shapes
:	�
Read_32/DisableCopyOnReadDisableCopyOnReadVread_32_disablecopyonread_transformer_encoder_4_encoder_2nd_normalizationlayer_2_gamma"/device:CPU:0*
_output_shapes
 �
Read_32/ReadVariableOpReadVariableOpVread_32_disablecopyonread_transformer_encoder_4_encoder_2nd_normalizationlayer_2_gamma^Read_32/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	*
dtype0k
Identity_64IdentityRead_32/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	a
Identity_65IdentityIdentity_64:output:0"/device:CPU:0*
T0*
_output_shapes
:	�
Read_33/DisableCopyOnReadDisableCopyOnReadUread_33_disablecopyonread_transformer_encoder_4_encoder_2nd_normalizationlayer_2_beta"/device:CPU:0*
_output_shapes
 �
Read_33/ReadVariableOpReadVariableOpUread_33_disablecopyonread_transformer_encoder_4_encoder_2nd_normalizationlayer_2_beta^Read_33/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	*
dtype0k
Identity_66IdentityRead_33/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	a
Identity_67IdentityIdentity_66:output:0"/device:CPU:0*
T0*
_output_shapes
:	�
Read_34/DisableCopyOnReadDisableCopyOnReadSread_34_disablecopyonread_transformer_encoder_4_encoder_feedforwardlayer_1_2_kernel"/device:CPU:0*
_output_shapes
 �
Read_34/ReadVariableOpReadVariableOpSread_34_disablecopyonread_transformer_encoder_4_encoder_feedforwardlayer_1_2_kernel^Read_34/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:		*
dtype0o
Identity_68IdentityRead_34/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:		e
Identity_69IdentityIdentity_68:output:0"/device:CPU:0*
T0*
_output_shapes

:		�
Read_35/DisableCopyOnReadDisableCopyOnReadQread_35_disablecopyonread_transformer_encoder_4_encoder_feedforwardlayer_1_2_bias"/device:CPU:0*
_output_shapes
 �
Read_35/ReadVariableOpReadVariableOpQread_35_disablecopyonread_transformer_encoder_4_encoder_feedforwardlayer_1_2_bias^Read_35/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	*
dtype0k
Identity_70IdentityRead_35/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	a
Identity_71IdentityIdentity_70:output:0"/device:CPU:0*
T0*
_output_shapes
:	�
Read_36/DisableCopyOnReadDisableCopyOnReadSread_36_disablecopyonread_transformer_encoder_4_encoder_feedforwardlayer_2_2_kernel"/device:CPU:0*
_output_shapes
 �
Read_36/ReadVariableOpReadVariableOpSread_36_disablecopyonread_transformer_encoder_4_encoder_feedforwardlayer_2_2_kernel^Read_36/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:		*
dtype0o
Identity_72IdentityRead_36/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:		e
Identity_73IdentityIdentity_72:output:0"/device:CPU:0*
T0*
_output_shapes

:		�
Read_37/DisableCopyOnReadDisableCopyOnReadQread_37_disablecopyonread_transformer_encoder_4_encoder_feedforwardlayer_2_2_bias"/device:CPU:0*
_output_shapes
 �
Read_37/ReadVariableOpReadVariableOpQread_37_disablecopyonread_transformer_encoder_4_encoder_feedforwardlayer_2_2_bias^Read_37/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	*
dtype0k
Identity_74IdentityRead_37/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	a
Identity_75IdentityIdentity_74:output:0"/device:CPU:0*
T0*
_output_shapes
:	�
Read_38/DisableCopyOnReadDisableCopyOnReadYread_38_disablecopyonread_transformer_encoder_5_encoder_selfattentionlayer_3_query_kernel"/device:CPU:0*
_output_shapes
 �
Read_38/ReadVariableOpReadVariableOpYread_38_disablecopyonread_transformer_encoder_5_encoder_selfattentionlayer_3_query_kernel^Read_38/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
:	*
dtype0s
Identity_76IdentityRead_38/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
:	i
Identity_77IdentityIdentity_76:output:0"/device:CPU:0*
T0*"
_output_shapes
:	�
Read_39/DisableCopyOnReadDisableCopyOnReadWread_39_disablecopyonread_transformer_encoder_5_encoder_selfattentionlayer_3_query_bias"/device:CPU:0*
_output_shapes
 �
Read_39/ReadVariableOpReadVariableOpWread_39_disablecopyonread_transformer_encoder_5_encoder_selfattentionlayer_3_query_bias^Read_39/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0o
Identity_78IdentityRead_39/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:e
Identity_79IdentityIdentity_78:output:0"/device:CPU:0*
T0*
_output_shapes

:�
Read_40/DisableCopyOnReadDisableCopyOnReadWread_40_disablecopyonread_transformer_encoder_5_encoder_selfattentionlayer_3_key_kernel"/device:CPU:0*
_output_shapes
 �
Read_40/ReadVariableOpReadVariableOpWread_40_disablecopyonread_transformer_encoder_5_encoder_selfattentionlayer_3_key_kernel^Read_40/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
:	*
dtype0s
Identity_80IdentityRead_40/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
:	i
Identity_81IdentityIdentity_80:output:0"/device:CPU:0*
T0*"
_output_shapes
:	�
Read_41/DisableCopyOnReadDisableCopyOnReadUread_41_disablecopyonread_transformer_encoder_5_encoder_selfattentionlayer_3_key_bias"/device:CPU:0*
_output_shapes
 �
Read_41/ReadVariableOpReadVariableOpUread_41_disablecopyonread_transformer_encoder_5_encoder_selfattentionlayer_3_key_bias^Read_41/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0o
Identity_82IdentityRead_41/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:e
Identity_83IdentityIdentity_82:output:0"/device:CPU:0*
T0*
_output_shapes

:�
Read_42/DisableCopyOnReadDisableCopyOnReadYread_42_disablecopyonread_transformer_encoder_5_encoder_selfattentionlayer_3_value_kernel"/device:CPU:0*
_output_shapes
 �
Read_42/ReadVariableOpReadVariableOpYread_42_disablecopyonread_transformer_encoder_5_encoder_selfattentionlayer_3_value_kernel^Read_42/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
:	*
dtype0s
Identity_84IdentityRead_42/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
:	i
Identity_85IdentityIdentity_84:output:0"/device:CPU:0*
T0*"
_output_shapes
:	�
Read_43/DisableCopyOnReadDisableCopyOnReadWread_43_disablecopyonread_transformer_encoder_5_encoder_selfattentionlayer_3_value_bias"/device:CPU:0*
_output_shapes
 �
Read_43/ReadVariableOpReadVariableOpWread_43_disablecopyonread_transformer_encoder_5_encoder_selfattentionlayer_3_value_bias^Read_43/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0o
Identity_86IdentityRead_43/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:e
Identity_87IdentityIdentity_86:output:0"/device:CPU:0*
T0*
_output_shapes

:�
Read_44/DisableCopyOnReadDisableCopyOnReaddread_44_disablecopyonread_transformer_encoder_5_encoder_selfattentionlayer_3_attention_output_kernel"/device:CPU:0*
_output_shapes
 �
Read_44/ReadVariableOpReadVariableOpdread_44_disablecopyonread_transformer_encoder_5_encoder_selfattentionlayer_3_attention_output_kernel^Read_44/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
:	*
dtype0s
Identity_88IdentityRead_44/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
:	i
Identity_89IdentityIdentity_88:output:0"/device:CPU:0*
T0*"
_output_shapes
:	�
Read_45/DisableCopyOnReadDisableCopyOnReadbread_45_disablecopyonread_transformer_encoder_5_encoder_selfattentionlayer_3_attention_output_bias"/device:CPU:0*
_output_shapes
 �
Read_45/ReadVariableOpReadVariableOpbread_45_disablecopyonread_transformer_encoder_5_encoder_selfattentionlayer_3_attention_output_bias^Read_45/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	*
dtype0k
Identity_90IdentityRead_45/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	a
Identity_91IdentityIdentity_90:output:0"/device:CPU:0*
T0*
_output_shapes
:	�
Read_46/DisableCopyOnReadDisableCopyOnReadVread_46_disablecopyonread_transformer_encoder_5_encoder_1st_normalizationlayer_3_gamma"/device:CPU:0*
_output_shapes
 �
Read_46/ReadVariableOpReadVariableOpVread_46_disablecopyonread_transformer_encoder_5_encoder_1st_normalizationlayer_3_gamma^Read_46/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	*
dtype0k
Identity_92IdentityRead_46/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	a
Identity_93IdentityIdentity_92:output:0"/device:CPU:0*
T0*
_output_shapes
:	�
Read_47/DisableCopyOnReadDisableCopyOnReadUread_47_disablecopyonread_transformer_encoder_5_encoder_1st_normalizationlayer_3_beta"/device:CPU:0*
_output_shapes
 �
Read_47/ReadVariableOpReadVariableOpUread_47_disablecopyonread_transformer_encoder_5_encoder_1st_normalizationlayer_3_beta^Read_47/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	*
dtype0k
Identity_94IdentityRead_47/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	a
Identity_95IdentityIdentity_94:output:0"/device:CPU:0*
T0*
_output_shapes
:	�
Read_48/DisableCopyOnReadDisableCopyOnReadVread_48_disablecopyonread_transformer_encoder_5_encoder_2nd_normalizationlayer_3_gamma"/device:CPU:0*
_output_shapes
 �
Read_48/ReadVariableOpReadVariableOpVread_48_disablecopyonread_transformer_encoder_5_encoder_2nd_normalizationlayer_3_gamma^Read_48/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	*
dtype0k
Identity_96IdentityRead_48/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	a
Identity_97IdentityIdentity_96:output:0"/device:CPU:0*
T0*
_output_shapes
:	�
Read_49/DisableCopyOnReadDisableCopyOnReadUread_49_disablecopyonread_transformer_encoder_5_encoder_2nd_normalizationlayer_3_beta"/device:CPU:0*
_output_shapes
 �
Read_49/ReadVariableOpReadVariableOpUread_49_disablecopyonread_transformer_encoder_5_encoder_2nd_normalizationlayer_3_beta^Read_49/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	*
dtype0k
Identity_98IdentityRead_49/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	a
Identity_99IdentityIdentity_98:output:0"/device:CPU:0*
T0*
_output_shapes
:	�
Read_50/DisableCopyOnReadDisableCopyOnReadSread_50_disablecopyonread_transformer_encoder_5_encoder_feedforwardlayer_1_3_kernel"/device:CPU:0*
_output_shapes
 �
Read_50/ReadVariableOpReadVariableOpSread_50_disablecopyonread_transformer_encoder_5_encoder_feedforwardlayer_1_3_kernel^Read_50/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:		*
dtype0p
Identity_100IdentityRead_50/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:		g
Identity_101IdentityIdentity_100:output:0"/device:CPU:0*
T0*
_output_shapes

:		�
Read_51/DisableCopyOnReadDisableCopyOnReadQread_51_disablecopyonread_transformer_encoder_5_encoder_feedforwardlayer_1_3_bias"/device:CPU:0*
_output_shapes
 �
Read_51/ReadVariableOpReadVariableOpQread_51_disablecopyonread_transformer_encoder_5_encoder_feedforwardlayer_1_3_bias^Read_51/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	*
dtype0l
Identity_102IdentityRead_51/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	c
Identity_103IdentityIdentity_102:output:0"/device:CPU:0*
T0*
_output_shapes
:	�
Read_52/DisableCopyOnReadDisableCopyOnReadSread_52_disablecopyonread_transformer_encoder_5_encoder_feedforwardlayer_2_3_kernel"/device:CPU:0*
_output_shapes
 �
Read_52/ReadVariableOpReadVariableOpSread_52_disablecopyonread_transformer_encoder_5_encoder_feedforwardlayer_2_3_kernel^Read_52/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:		*
dtype0p
Identity_104IdentityRead_52/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:		g
Identity_105IdentityIdentity_104:output:0"/device:CPU:0*
T0*
_output_shapes

:		�
Read_53/DisableCopyOnReadDisableCopyOnReadQread_53_disablecopyonread_transformer_encoder_5_encoder_feedforwardlayer_2_3_bias"/device:CPU:0*
_output_shapes
 �
Read_53/ReadVariableOpReadVariableOpQread_53_disablecopyonread_transformer_encoder_5_encoder_feedforwardlayer_2_3_bias^Read_53/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	*
dtype0l
Identity_106IdentityRead_53/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	c
Identity_107IdentityIdentity_106:output:0"/device:CPU:0*
T0*
_output_shapes
:	�
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:7*
dtype0*�
value�B�7B5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB'variables/26/.ATTRIBUTES/VARIABLE_VALUEB'variables/27/.ATTRIBUTES/VARIABLE_VALUEB'variables/28/.ATTRIBUTES/VARIABLE_VALUEB'variables/29/.ATTRIBUTES/VARIABLE_VALUEB'variables/30/.ATTRIBUTES/VARIABLE_VALUEB'variables/31/.ATTRIBUTES/VARIABLE_VALUEB'variables/32/.ATTRIBUTES/VARIABLE_VALUEB'variables/33/.ATTRIBUTES/VARIABLE_VALUEB'variables/34/.ATTRIBUTES/VARIABLE_VALUEB'variables/35/.ATTRIBUTES/VARIABLE_VALUEB'variables/36/.ATTRIBUTES/VARIABLE_VALUEB'variables/37/.ATTRIBUTES/VARIABLE_VALUEB'variables/38/.ATTRIBUTES/VARIABLE_VALUEB'variables/39/.ATTRIBUTES/VARIABLE_VALUEB'variables/40/.ATTRIBUTES/VARIABLE_VALUEB'variables/41/.ATTRIBUTES/VARIABLE_VALUEB'variables/42/.ATTRIBUTES/VARIABLE_VALUEB'variables/43/.ATTRIBUTES/VARIABLE_VALUEB'variables/44/.ATTRIBUTES/VARIABLE_VALUEB'variables/45/.ATTRIBUTES/VARIABLE_VALUEB'variables/46/.ATTRIBUTES/VARIABLE_VALUEB'variables/47/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:7*
dtype0*�
valuexBv7B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Identity_1:output:0Identity_3:output:0Identity_5:output:0Identity_7:output:0Identity_9:output:0Identity_11:output:0Identity_13:output:0Identity_15:output:0Identity_17:output:0Identity_19:output:0Identity_21:output:0Identity_23:output:0Identity_25:output:0Identity_27:output:0Identity_29:output:0Identity_31:output:0Identity_33:output:0Identity_35:output:0Identity_37:output:0Identity_39:output:0Identity_41:output:0Identity_43:output:0Identity_45:output:0Identity_47:output:0Identity_49:output:0Identity_51:output:0Identity_53:output:0Identity_55:output:0Identity_57:output:0Identity_59:output:0Identity_61:output:0Identity_63:output:0Identity_65:output:0Identity_67:output:0Identity_69:output:0Identity_71:output:0Identity_73:output:0Identity_75:output:0Identity_77:output:0Identity_79:output:0Identity_81:output:0Identity_83:output:0Identity_85:output:0Identity_87:output:0Identity_89:output:0Identity_91:output:0Identity_93:output:0Identity_95:output:0Identity_97:output:0Identity_99:output:0Identity_101:output:0Identity_103:output:0Identity_105:output:0Identity_107:output:0savev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *E
dtypes;
927�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 j
Identity_108Identityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: W
Identity_109IdentityIdentity_108:output:0^NoOp*
T0*
_output_shapes
: �
NoOpNoOp^MergeV2Checkpoints^Read/DisableCopyOnRead^Read/ReadVariableOp^Read_1/DisableCopyOnRead^Read_1/ReadVariableOp^Read_10/DisableCopyOnRead^Read_10/ReadVariableOp^Read_11/DisableCopyOnRead^Read_11/ReadVariableOp^Read_12/DisableCopyOnRead^Read_12/ReadVariableOp^Read_13/DisableCopyOnRead^Read_13/ReadVariableOp^Read_14/DisableCopyOnRead^Read_14/ReadVariableOp^Read_15/DisableCopyOnRead^Read_15/ReadVariableOp^Read_16/DisableCopyOnRead^Read_16/ReadVariableOp^Read_17/DisableCopyOnRead^Read_17/ReadVariableOp^Read_18/DisableCopyOnRead^Read_18/ReadVariableOp^Read_19/DisableCopyOnRead^Read_19/ReadVariableOp^Read_2/DisableCopyOnRead^Read_2/ReadVariableOp^Read_20/DisableCopyOnRead^Read_20/ReadVariableOp^Read_21/DisableCopyOnRead^Read_21/ReadVariableOp^Read_22/DisableCopyOnRead^Read_22/ReadVariableOp^Read_23/DisableCopyOnRead^Read_23/ReadVariableOp^Read_24/DisableCopyOnRead^Read_24/ReadVariableOp^Read_25/DisableCopyOnRead^Read_25/ReadVariableOp^Read_26/DisableCopyOnRead^Read_26/ReadVariableOp^Read_27/DisableCopyOnRead^Read_27/ReadVariableOp^Read_28/DisableCopyOnRead^Read_28/ReadVariableOp^Read_29/DisableCopyOnRead^Read_29/ReadVariableOp^Read_3/DisableCopyOnRead^Read_3/ReadVariableOp^Read_30/DisableCopyOnRead^Read_30/ReadVariableOp^Read_31/DisableCopyOnRead^Read_31/ReadVariableOp^Read_32/DisableCopyOnRead^Read_32/ReadVariableOp^Read_33/DisableCopyOnRead^Read_33/ReadVariableOp^Read_34/DisableCopyOnRead^Read_34/ReadVariableOp^Read_35/DisableCopyOnRead^Read_35/ReadVariableOp^Read_36/DisableCopyOnRead^Read_36/ReadVariableOp^Read_37/DisableCopyOnRead^Read_37/ReadVariableOp^Read_38/DisableCopyOnRead^Read_38/ReadVariableOp^Read_39/DisableCopyOnRead^Read_39/ReadVariableOp^Read_4/DisableCopyOnRead^Read_4/ReadVariableOp^Read_40/DisableCopyOnRead^Read_40/ReadVariableOp^Read_41/DisableCopyOnRead^Read_41/ReadVariableOp^Read_42/DisableCopyOnRead^Read_42/ReadVariableOp^Read_43/DisableCopyOnRead^Read_43/ReadVariableOp^Read_44/DisableCopyOnRead^Read_44/ReadVariableOp^Read_45/DisableCopyOnRead^Read_45/ReadVariableOp^Read_46/DisableCopyOnRead^Read_46/ReadVariableOp^Read_47/DisableCopyOnRead^Read_47/ReadVariableOp^Read_48/DisableCopyOnRead^Read_48/ReadVariableOp^Read_49/DisableCopyOnRead^Read_49/ReadVariableOp^Read_5/DisableCopyOnRead^Read_5/ReadVariableOp^Read_50/DisableCopyOnRead^Read_50/ReadVariableOp^Read_51/DisableCopyOnRead^Read_51/ReadVariableOp^Read_52/DisableCopyOnRead^Read_52/ReadVariableOp^Read_53/DisableCopyOnRead^Read_53/ReadVariableOp^Read_6/DisableCopyOnRead^Read_6/ReadVariableOp^Read_7/DisableCopyOnRead^Read_7/ReadVariableOp^Read_8/DisableCopyOnRead^Read_8/ReadVariableOp^Read_9/DisableCopyOnRead^Read_9/ReadVariableOp*
_output_shapes
 "%
identity_109Identity_109:output:0*(
_construction_contextkEagerRuntime*�
_input_shapesr
p: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints20
Read/DisableCopyOnReadRead/DisableCopyOnRead2*
Read/ReadVariableOpRead/ReadVariableOp24
Read_1/DisableCopyOnReadRead_1/DisableCopyOnRead2.
Read_1/ReadVariableOpRead_1/ReadVariableOp26
Read_10/DisableCopyOnReadRead_10/DisableCopyOnRead20
Read_10/ReadVariableOpRead_10/ReadVariableOp26
Read_11/DisableCopyOnReadRead_11/DisableCopyOnRead20
Read_11/ReadVariableOpRead_11/ReadVariableOp26
Read_12/DisableCopyOnReadRead_12/DisableCopyOnRead20
Read_12/ReadVariableOpRead_12/ReadVariableOp26
Read_13/DisableCopyOnReadRead_13/DisableCopyOnRead20
Read_13/ReadVariableOpRead_13/ReadVariableOp26
Read_14/DisableCopyOnReadRead_14/DisableCopyOnRead20
Read_14/ReadVariableOpRead_14/ReadVariableOp26
Read_15/DisableCopyOnReadRead_15/DisableCopyOnRead20
Read_15/ReadVariableOpRead_15/ReadVariableOp26
Read_16/DisableCopyOnReadRead_16/DisableCopyOnRead20
Read_16/ReadVariableOpRead_16/ReadVariableOp26
Read_17/DisableCopyOnReadRead_17/DisableCopyOnRead20
Read_17/ReadVariableOpRead_17/ReadVariableOp26
Read_18/DisableCopyOnReadRead_18/DisableCopyOnRead20
Read_18/ReadVariableOpRead_18/ReadVariableOp26
Read_19/DisableCopyOnReadRead_19/DisableCopyOnRead20
Read_19/ReadVariableOpRead_19/ReadVariableOp24
Read_2/DisableCopyOnReadRead_2/DisableCopyOnRead2.
Read_2/ReadVariableOpRead_2/ReadVariableOp26
Read_20/DisableCopyOnReadRead_20/DisableCopyOnRead20
Read_20/ReadVariableOpRead_20/ReadVariableOp26
Read_21/DisableCopyOnReadRead_21/DisableCopyOnRead20
Read_21/ReadVariableOpRead_21/ReadVariableOp26
Read_22/DisableCopyOnReadRead_22/DisableCopyOnRead20
Read_22/ReadVariableOpRead_22/ReadVariableOp26
Read_23/DisableCopyOnReadRead_23/DisableCopyOnRead20
Read_23/ReadVariableOpRead_23/ReadVariableOp26
Read_24/DisableCopyOnReadRead_24/DisableCopyOnRead20
Read_24/ReadVariableOpRead_24/ReadVariableOp26
Read_25/DisableCopyOnReadRead_25/DisableCopyOnRead20
Read_25/ReadVariableOpRead_25/ReadVariableOp26
Read_26/DisableCopyOnReadRead_26/DisableCopyOnRead20
Read_26/ReadVariableOpRead_26/ReadVariableOp26
Read_27/DisableCopyOnReadRead_27/DisableCopyOnRead20
Read_27/ReadVariableOpRead_27/ReadVariableOp26
Read_28/DisableCopyOnReadRead_28/DisableCopyOnRead20
Read_28/ReadVariableOpRead_28/ReadVariableOp26
Read_29/DisableCopyOnReadRead_29/DisableCopyOnRead20
Read_29/ReadVariableOpRead_29/ReadVariableOp24
Read_3/DisableCopyOnReadRead_3/DisableCopyOnRead2.
Read_3/ReadVariableOpRead_3/ReadVariableOp26
Read_30/DisableCopyOnReadRead_30/DisableCopyOnRead20
Read_30/ReadVariableOpRead_30/ReadVariableOp26
Read_31/DisableCopyOnReadRead_31/DisableCopyOnRead20
Read_31/ReadVariableOpRead_31/ReadVariableOp26
Read_32/DisableCopyOnReadRead_32/DisableCopyOnRead20
Read_32/ReadVariableOpRead_32/ReadVariableOp26
Read_33/DisableCopyOnReadRead_33/DisableCopyOnRead20
Read_33/ReadVariableOpRead_33/ReadVariableOp26
Read_34/DisableCopyOnReadRead_34/DisableCopyOnRead20
Read_34/ReadVariableOpRead_34/ReadVariableOp26
Read_35/DisableCopyOnReadRead_35/DisableCopyOnRead20
Read_35/ReadVariableOpRead_35/ReadVariableOp26
Read_36/DisableCopyOnReadRead_36/DisableCopyOnRead20
Read_36/ReadVariableOpRead_36/ReadVariableOp26
Read_37/DisableCopyOnReadRead_37/DisableCopyOnRead20
Read_37/ReadVariableOpRead_37/ReadVariableOp26
Read_38/DisableCopyOnReadRead_38/DisableCopyOnRead20
Read_38/ReadVariableOpRead_38/ReadVariableOp26
Read_39/DisableCopyOnReadRead_39/DisableCopyOnRead20
Read_39/ReadVariableOpRead_39/ReadVariableOp24
Read_4/DisableCopyOnReadRead_4/DisableCopyOnRead2.
Read_4/ReadVariableOpRead_4/ReadVariableOp26
Read_40/DisableCopyOnReadRead_40/DisableCopyOnRead20
Read_40/ReadVariableOpRead_40/ReadVariableOp26
Read_41/DisableCopyOnReadRead_41/DisableCopyOnRead20
Read_41/ReadVariableOpRead_41/ReadVariableOp26
Read_42/DisableCopyOnReadRead_42/DisableCopyOnRead20
Read_42/ReadVariableOpRead_42/ReadVariableOp26
Read_43/DisableCopyOnReadRead_43/DisableCopyOnRead20
Read_43/ReadVariableOpRead_43/ReadVariableOp26
Read_44/DisableCopyOnReadRead_44/DisableCopyOnRead20
Read_44/ReadVariableOpRead_44/ReadVariableOp26
Read_45/DisableCopyOnReadRead_45/DisableCopyOnRead20
Read_45/ReadVariableOpRead_45/ReadVariableOp26
Read_46/DisableCopyOnReadRead_46/DisableCopyOnRead20
Read_46/ReadVariableOpRead_46/ReadVariableOp26
Read_47/DisableCopyOnReadRead_47/DisableCopyOnRead20
Read_47/ReadVariableOpRead_47/ReadVariableOp26
Read_48/DisableCopyOnReadRead_48/DisableCopyOnRead20
Read_48/ReadVariableOpRead_48/ReadVariableOp26
Read_49/DisableCopyOnReadRead_49/DisableCopyOnRead20
Read_49/ReadVariableOpRead_49/ReadVariableOp24
Read_5/DisableCopyOnReadRead_5/DisableCopyOnRead2.
Read_5/ReadVariableOpRead_5/ReadVariableOp26
Read_50/DisableCopyOnReadRead_50/DisableCopyOnRead20
Read_50/ReadVariableOpRead_50/ReadVariableOp26
Read_51/DisableCopyOnReadRead_51/DisableCopyOnRead20
Read_51/ReadVariableOpRead_51/ReadVariableOp26
Read_52/DisableCopyOnReadRead_52/DisableCopyOnRead20
Read_52/ReadVariableOpRead_52/ReadVariableOp26
Read_53/DisableCopyOnReadRead_53/DisableCopyOnRead20
Read_53/ReadVariableOpRead_53/ReadVariableOp24
Read_6/DisableCopyOnReadRead_6/DisableCopyOnRead2.
Read_6/ReadVariableOpRead_6/ReadVariableOp24
Read_7/DisableCopyOnReadRead_7/DisableCopyOnRead2.
Read_7/ReadVariableOpRead_7/ReadVariableOp24
Read_8/DisableCopyOnReadRead_8/DisableCopyOnRead2.
Read_8/ReadVariableOpRead_8/ReadVariableOp24
Read_9/DisableCopyOnReadRead_9/DisableCopyOnRead2.
Read_9/ReadVariableOpRead_9/ReadVariableOp:=79

_output_shapes
: 

_user_specified_nameConst:W6S
Q
_user_specified_name97transformer_encoder_5/Encoder-FeedForwardLayer_2_3/bias:Y5U
S
_user_specified_name;9transformer_encoder_5/Encoder-FeedForwardLayer_2_3/kernel:W4S
Q
_user_specified_name97transformer_encoder_5/Encoder-FeedForwardLayer_1_3/bias:Y3U
S
_user_specified_name;9transformer_encoder_5/Encoder-FeedForwardLayer_1_3/kernel:[2W
U
_user_specified_name=;transformer_encoder_5/Encoder-2nd-NormalizationLayer-3/beta:\1X
V
_user_specified_name><transformer_encoder_5/Encoder-2nd-NormalizationLayer-3/gamma:[0W
U
_user_specified_name=;transformer_encoder_5/Encoder-1st-NormalizationLayer-3/beta:\/X
V
_user_specified_name><transformer_encoder_5/Encoder-1st-NormalizationLayer-3/gamma:h.d
b
_user_specified_nameJHtransformer_encoder_5/Encoder-SelfAttentionLayer-3/attention_output/bias:j-f
d
_user_specified_nameLJtransformer_encoder_5/Encoder-SelfAttentionLayer-3/attention_output/kernel:],Y
W
_user_specified_name?=transformer_encoder_5/Encoder-SelfAttentionLayer-3/value/bias:_+[
Y
_user_specified_nameA?transformer_encoder_5/Encoder-SelfAttentionLayer-3/value/kernel:[*W
U
_user_specified_name=;transformer_encoder_5/Encoder-SelfAttentionLayer-3/key/bias:])Y
W
_user_specified_name?=transformer_encoder_5/Encoder-SelfAttentionLayer-3/key/kernel:](Y
W
_user_specified_name?=transformer_encoder_5/Encoder-SelfAttentionLayer-3/query/bias:_'[
Y
_user_specified_nameA?transformer_encoder_5/Encoder-SelfAttentionLayer-3/query/kernel:W&S
Q
_user_specified_name97transformer_encoder_4/Encoder-FeedForwardLayer_2_2/bias:Y%U
S
_user_specified_name;9transformer_encoder_4/Encoder-FeedForwardLayer_2_2/kernel:W$S
Q
_user_specified_name97transformer_encoder_4/Encoder-FeedForwardLayer_1_2/bias:Y#U
S
_user_specified_name;9transformer_encoder_4/Encoder-FeedForwardLayer_1_2/kernel:["W
U
_user_specified_name=;transformer_encoder_4/Encoder-2nd-NormalizationLayer-2/beta:\!X
V
_user_specified_name><transformer_encoder_4/Encoder-2nd-NormalizationLayer-2/gamma:[ W
U
_user_specified_name=;transformer_encoder_4/Encoder-1st-NormalizationLayer-2/beta:\X
V
_user_specified_name><transformer_encoder_4/Encoder-1st-NormalizationLayer-2/gamma:hd
b
_user_specified_nameJHtransformer_encoder_4/Encoder-SelfAttentionLayer-2/attention_output/bias:jf
d
_user_specified_nameLJtransformer_encoder_4/Encoder-SelfAttentionLayer-2/attention_output/kernel:]Y
W
_user_specified_name?=transformer_encoder_4/Encoder-SelfAttentionLayer-2/value/bias:_[
Y
_user_specified_nameA?transformer_encoder_4/Encoder-SelfAttentionLayer-2/value/kernel:[W
U
_user_specified_name=;transformer_encoder_4/Encoder-SelfAttentionLayer-2/key/bias:]Y
W
_user_specified_name?=transformer_encoder_4/Encoder-SelfAttentionLayer-2/key/kernel:]Y
W
_user_specified_name?=transformer_encoder_4/Encoder-SelfAttentionLayer-2/query/bias:_[
Y
_user_specified_nameA?transformer_encoder_4/Encoder-SelfAttentionLayer-2/query/kernel:WS
Q
_user_specified_name97transformer_encoder_3/Encoder-FeedForwardLayer_2_1/bias:YU
S
_user_specified_name;9transformer_encoder_3/Encoder-FeedForwardLayer_2_1/kernel:WS
Q
_user_specified_name97transformer_encoder_3/Encoder-FeedForwardLayer_1_1/bias:YU
S
_user_specified_name;9transformer_encoder_3/Encoder-FeedForwardLayer_1_1/kernel:[W
U
_user_specified_name=;transformer_encoder_3/Encoder-2nd-NormalizationLayer-1/beta:\X
V
_user_specified_name><transformer_encoder_3/Encoder-2nd-NormalizationLayer-1/gamma:[W
U
_user_specified_name=;transformer_encoder_3/Encoder-1st-NormalizationLayer-1/beta:\X
V
_user_specified_name><transformer_encoder_3/Encoder-1st-NormalizationLayer-1/gamma:hd
b
_user_specified_nameJHtransformer_encoder_3/Encoder-SelfAttentionLayer-1/attention_output/bias:jf
d
_user_specified_nameLJtransformer_encoder_3/Encoder-SelfAttentionLayer-1/attention_output/kernel:]Y
W
_user_specified_name?=transformer_encoder_3/Encoder-SelfAttentionLayer-1/value/bias:_[
Y
_user_specified_nameA?transformer_encoder_3/Encoder-SelfAttentionLayer-1/value/kernel:[
W
U
_user_specified_name=;transformer_encoder_3/Encoder-SelfAttentionLayer-1/key/bias:]	Y
W
_user_specified_name?=transformer_encoder_3/Encoder-SelfAttentionLayer-1/key/kernel:]Y
W
_user_specified_name?=transformer_encoder_3/Encoder-SelfAttentionLayer-1/query/bias:_[
Y
_user_specified_nameA?transformer_encoder_3/Encoder-SelfAttentionLayer-1/query/kernel::6
4
_user_specified_namePredictionImprovement/bias:<8
6
_user_specified_namePredictionImprovement/kernel:C?
=
_user_specified_name%#FullyConnectedLayerImprovement/bias:EA
?
_user_specified_name'%FullyConnectedLayerImprovement/kernel:3/
-
_user_specified_nameFinalLayerNorm/beta:40
.
_user_specified_nameFinalLayerNorm/gamma:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
x
\__inference_ReduceStackDimensionViaSummation_layer_call_and_return_conditional_losses_230490

inputs
identityW
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :d
SumSuminputsSum/reduction_indices:output:0*
T0*'
_output_shapes
:���������	N
	truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �Bf
truedivRealDivSum:output:0truediv/y:output:0*
T0*'
_output_shapes
:���������	S
IdentityIdentitytruediv:z:0*
T0*'
_output_shapes
:���������	"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������P	:S O
+
_output_shapes
:���������P	
 
_user_specified_nameinputs
�
l
P__inference_StandardizeTimeLimit_layer_call_and_return_conditional_losses_233304

inputs
identityJ
sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  pAT
subSubinputssub/y:output:0*
T0*'
_output_shapes
:���������N
	truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �@a
truedivRealDivsub:z:0truediv/y:output:0*
T0*'
_output_shapes
:���������S
IdentityIdentitytruediv:z:0*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
^
B__inference_Output_layer_call_and_return_conditional_losses_231225

inputs
identity=
SqueezeSqueezeinputs*
T0*
_output_shapes
:I
IdentityIdentitySqueeze:output:0*
T0*
_output_shapes
:"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
x
\__inference_ReduceStackDimensionViaSummation_layer_call_and_return_conditional_losses_231198

inputs
identityW
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :d
SumSuminputsSum/reduction_indices:output:0*
T0*'
_output_shapes
:���������	N
	truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �Bf
truedivRealDivSum:output:0truediv/y:output:0*
T0*'
_output_shapes
:���������	S
IdentityIdentitytruediv:z:0*
T0*'
_output_shapes
:���������	"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������P	:S O
+
_output_shapes
:���������P	
 
_user_specified_nameinputs
�
x
\__inference_ReduceStackDimensionViaSummation_layer_call_and_return_conditional_losses_233278

inputs
identityW
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :d
SumSuminputsSum/reduction_indices:output:0*
T0*'
_output_shapes
:���������	N
	truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �Bf
truedivRealDivSum:output:0truediv/y:output:0*
T0*'
_output_shapes
:���������	S
IdentityIdentitytruediv:z:0*
T0*'
_output_shapes
:���������	"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������P	:S O
+
_output_shapes
:���������P	
 
_user_specified_nameinputs
��
�
Q__inference_transformer_encoder_3_layer_call_and_return_conditional_losses_232329

inputsJ
<encoder_1st_normalizationlayer_1_mul_readvariableop_resource:	J
<encoder_1st_normalizationlayer_1_add_readvariableop_resource:	^
Hencoder_selfattentionlayer_1_query_einsum_einsum_readvariableop_resource:	P
>encoder_selfattentionlayer_1_query_add_readvariableop_resource:\
Fencoder_selfattentionlayer_1_key_einsum_einsum_readvariableop_resource:	N
<encoder_selfattentionlayer_1_key_add_readvariableop_resource:^
Hencoder_selfattentionlayer_1_value_einsum_einsum_readvariableop_resource:	P
>encoder_selfattentionlayer_1_value_add_readvariableop_resource:i
Sencoder_selfattentionlayer_1_attention_output_einsum_einsum_readvariableop_resource:	W
Iencoder_selfattentionlayer_1_attention_output_add_readvariableop_resource:	J
<encoder_2nd_normalizationlayer_1_mul_readvariableop_resource:	J
<encoder_2nd_normalizationlayer_1_add_readvariableop_resource:	P
>encoder_feedforwardlayer_1_1_tensordot_readvariableop_resource:		J
<encoder_feedforwardlayer_1_1_biasadd_readvariableop_resource:	P
>encoder_feedforwardlayer_2_1_tensordot_readvariableop_resource:		J
<encoder_feedforwardlayer_2_1_biasadd_readvariableop_resource:	
identity

identity_1��3Encoder-1st-NormalizationLayer-1/add/ReadVariableOp�3Encoder-1st-NormalizationLayer-1/mul/ReadVariableOp�3Encoder-2nd-NormalizationLayer-1/add/ReadVariableOp�3Encoder-2nd-NormalizationLayer-1/mul/ReadVariableOp�3Encoder-FeedForwardLayer_1_1/BiasAdd/ReadVariableOp�5Encoder-FeedForwardLayer_1_1/Tensordot/ReadVariableOp�3Encoder-FeedForwardLayer_2_1/BiasAdd/ReadVariableOp�5Encoder-FeedForwardLayer_2_1/Tensordot/ReadVariableOp�@Encoder-SelfAttentionLayer-1/attention_output/add/ReadVariableOp�JEncoder-SelfAttentionLayer-1/attention_output/einsum/Einsum/ReadVariableOp�3Encoder-SelfAttentionLayer-1/key/add/ReadVariableOp�=Encoder-SelfAttentionLayer-1/key/einsum/Einsum/ReadVariableOp�5Encoder-SelfAttentionLayer-1/query/add/ReadVariableOp�?Encoder-SelfAttentionLayer-1/query/einsum/Einsum/ReadVariableOp�5Encoder-SelfAttentionLayer-1/value/add/ReadVariableOp�?Encoder-SelfAttentionLayer-1/value/einsum/Einsum/ReadVariableOpj
&Encoder-1st-NormalizationLayer-1/ShapeShapeinputs*
T0*
_output_shapes
::��~
4Encoder-1st-NormalizationLayer-1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
6Encoder-1st-NormalizationLayer-1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
6Encoder-1st-NormalizationLayer-1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
.Encoder-1st-NormalizationLayer-1/strided_sliceStridedSlice/Encoder-1st-NormalizationLayer-1/Shape:output:0=Encoder-1st-NormalizationLayer-1/strided_slice/stack:output:0?Encoder-1st-NormalizationLayer-1/strided_slice/stack_1:output:0?Encoder-1st-NormalizationLayer-1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskp
&Encoder-1st-NormalizationLayer-1/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
%Encoder-1st-NormalizationLayer-1/ProdProd7Encoder-1st-NormalizationLayer-1/strided_slice:output:0/Encoder-1st-NormalizationLayer-1/Const:output:0*
T0*
_output_shapes
: �
6Encoder-1st-NormalizationLayer-1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:�
8Encoder-1st-NormalizationLayer-1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: �
8Encoder-1st-NormalizationLayer-1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
0Encoder-1st-NormalizationLayer-1/strided_slice_1StridedSlice/Encoder-1st-NormalizationLayer-1/Shape:output:0?Encoder-1st-NormalizationLayer-1/strided_slice_1/stack:output:0AEncoder-1st-NormalizationLayer-1/strided_slice_1/stack_1:output:0AEncoder-1st-NormalizationLayer-1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_maskr
(Encoder-1st-NormalizationLayer-1/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
'Encoder-1st-NormalizationLayer-1/Prod_1Prod9Encoder-1st-NormalizationLayer-1/strided_slice_1:output:01Encoder-1st-NormalizationLayer-1/Const_1:output:0*
T0*
_output_shapes
: r
0Encoder-1st-NormalizationLayer-1/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :r
0Encoder-1st-NormalizationLayer-1/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :�
.Encoder-1st-NormalizationLayer-1/Reshape/shapePack9Encoder-1st-NormalizationLayer-1/Reshape/shape/0:output:0.Encoder-1st-NormalizationLayer-1/Prod:output:00Encoder-1st-NormalizationLayer-1/Prod_1:output:09Encoder-1st-NormalizationLayer-1/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:�
(Encoder-1st-NormalizationLayer-1/ReshapeReshapeinputs7Encoder-1st-NormalizationLayer-1/Reshape/shape:output:0*
T0*8
_output_shapes&
$:"�������������������
,Encoder-1st-NormalizationLayer-1/ones/packedPack.Encoder-1st-NormalizationLayer-1/Prod:output:0*
N*
T0*
_output_shapes
:p
+Encoder-1st-NormalizationLayer-1/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
%Encoder-1st-NormalizationLayer-1/onesFill5Encoder-1st-NormalizationLayer-1/ones/packed:output:04Encoder-1st-NormalizationLayer-1/ones/Const:output:0*
T0*#
_output_shapes
:����������
-Encoder-1st-NormalizationLayer-1/zeros/packedPack.Encoder-1st-NormalizationLayer-1/Prod:output:0*
N*
T0*
_output_shapes
:q
,Encoder-1st-NormalizationLayer-1/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
&Encoder-1st-NormalizationLayer-1/zerosFill6Encoder-1st-NormalizationLayer-1/zeros/packed:output:05Encoder-1st-NormalizationLayer-1/zeros/Const:output:0*
T0*#
_output_shapes
:���������k
(Encoder-1st-NormalizationLayer-1/Const_2Const*
_output_shapes
: *
dtype0*
valueB k
(Encoder-1st-NormalizationLayer-1/Const_3Const*
_output_shapes
: *
dtype0*
valueB �
1Encoder-1st-NormalizationLayer-1/FusedBatchNormV3FusedBatchNormV31Encoder-1st-NormalizationLayer-1/Reshape:output:0.Encoder-1st-NormalizationLayer-1/ones:output:0/Encoder-1st-NormalizationLayer-1/zeros:output:01Encoder-1st-NormalizationLayer-1/Const_2:output:01Encoder-1st-NormalizationLayer-1/Const_3:output:0*
T0*
U0*x
_output_shapesf
d:"������������������:���������:���������:���������:���������:*
data_formatNCHW*
epsilon%o�:�
*Encoder-1st-NormalizationLayer-1/Reshape_1Reshape5Encoder-1st-NormalizationLayer-1/FusedBatchNormV3:y:0/Encoder-1st-NormalizationLayer-1/Shape:output:0*
T0*+
_output_shapes
:���������P	�
3Encoder-1st-NormalizationLayer-1/mul/ReadVariableOpReadVariableOp<encoder_1st_normalizationlayer_1_mul_readvariableop_resource*
_output_shapes
:	*
dtype0�
$Encoder-1st-NormalizationLayer-1/mulMul3Encoder-1st-NormalizationLayer-1/Reshape_1:output:0;Encoder-1st-NormalizationLayer-1/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������P	�
3Encoder-1st-NormalizationLayer-1/add/ReadVariableOpReadVariableOp<encoder_1st_normalizationlayer_1_add_readvariableop_resource*
_output_shapes
:	*
dtype0�
$Encoder-1st-NormalizationLayer-1/addAddV2(Encoder-1st-NormalizationLayer-1/mul:z:0;Encoder-1st-NormalizationLayer-1/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������P	�
?Encoder-SelfAttentionLayer-1/query/einsum/Einsum/ReadVariableOpReadVariableOpHencoder_selfattentionlayer_1_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:	*
dtype0�
0Encoder-SelfAttentionLayer-1/query/einsum/EinsumEinsum(Encoder-1st-NormalizationLayer-1/add:z:0GEncoder-SelfAttentionLayer-1/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������P*
equationabc,cde->abde�
5Encoder-SelfAttentionLayer-1/query/add/ReadVariableOpReadVariableOp>encoder_selfattentionlayer_1_query_add_readvariableop_resource*
_output_shapes

:*
dtype0�
&Encoder-SelfAttentionLayer-1/query/addAddV29Encoder-SelfAttentionLayer-1/query/einsum/Einsum:output:0=Encoder-SelfAttentionLayer-1/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������P�
=Encoder-SelfAttentionLayer-1/key/einsum/Einsum/ReadVariableOpReadVariableOpFencoder_selfattentionlayer_1_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:	*
dtype0�
.Encoder-SelfAttentionLayer-1/key/einsum/EinsumEinsum(Encoder-1st-NormalizationLayer-1/add:z:0EEncoder-SelfAttentionLayer-1/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������P*
equationabc,cde->abde�
3Encoder-SelfAttentionLayer-1/key/add/ReadVariableOpReadVariableOp<encoder_selfattentionlayer_1_key_add_readvariableop_resource*
_output_shapes

:*
dtype0�
$Encoder-SelfAttentionLayer-1/key/addAddV27Encoder-SelfAttentionLayer-1/key/einsum/Einsum:output:0;Encoder-SelfAttentionLayer-1/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������P�
?Encoder-SelfAttentionLayer-1/value/einsum/Einsum/ReadVariableOpReadVariableOpHencoder_selfattentionlayer_1_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:	*
dtype0�
0Encoder-SelfAttentionLayer-1/value/einsum/EinsumEinsum(Encoder-1st-NormalizationLayer-1/add:z:0GEncoder-SelfAttentionLayer-1/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������P*
equationabc,cde->abde�
5Encoder-SelfAttentionLayer-1/value/add/ReadVariableOpReadVariableOp>encoder_selfattentionlayer_1_value_add_readvariableop_resource*
_output_shapes

:*
dtype0�
&Encoder-SelfAttentionLayer-1/value/addAddV29Encoder-SelfAttentionLayer-1/value/einsum/Einsum:output:0=Encoder-SelfAttentionLayer-1/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������Pg
"Encoder-SelfAttentionLayer-1/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *:�?�
 Encoder-SelfAttentionLayer-1/MulMul*Encoder-SelfAttentionLayer-1/query/add:z:0+Encoder-SelfAttentionLayer-1/Mul/y:output:0*
T0*/
_output_shapes
:���������P�
*Encoder-SelfAttentionLayer-1/einsum/EinsumEinsum(Encoder-SelfAttentionLayer-1/key/add:z:0$Encoder-SelfAttentionLayer-1/Mul:z:0*
N*
T0*/
_output_shapes
:���������PP*
equationaecd,abcd->acbe�
,Encoder-SelfAttentionLayer-1/softmax/SoftmaxSoftmax3Encoder-SelfAttentionLayer-1/einsum/Einsum:output:0*
T0*/
_output_shapes
:���������PP�
-Encoder-SelfAttentionLayer-1/dropout/IdentityIdentity6Encoder-SelfAttentionLayer-1/softmax/Softmax:softmax:0*
T0*/
_output_shapes
:���������PP�
,Encoder-SelfAttentionLayer-1/einsum_1/EinsumEinsum6Encoder-SelfAttentionLayer-1/dropout/Identity:output:0*Encoder-SelfAttentionLayer-1/value/add:z:0*
N*
T0*/
_output_shapes
:���������P*
equationacbe,aecd->abcd�
JEncoder-SelfAttentionLayer-1/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpSencoder_selfattentionlayer_1_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:	*
dtype0�
;Encoder-SelfAttentionLayer-1/attention_output/einsum/EinsumEinsum5Encoder-SelfAttentionLayer-1/einsum_1/Einsum:output:0REncoder-SelfAttentionLayer-1/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:���������P	*
equationabcd,cde->abe�
@Encoder-SelfAttentionLayer-1/attention_output/add/ReadVariableOpReadVariableOpIencoder_selfattentionlayer_1_attention_output_add_readvariableop_resource*
_output_shapes
:	*
dtype0�
1Encoder-SelfAttentionLayer-1/attention_output/addAddV2DEncoder-SelfAttentionLayer-1/attention_output/einsum/Einsum:output:0HEncoder-SelfAttentionLayer-1/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������P	�
Encoder-1st-AdditionLayer-1/addAddV2inputs5Encoder-SelfAttentionLayer-1/attention_output/add:z:0*
T0*+
_output_shapes
:���������P	�
&Encoder-2nd-NormalizationLayer-1/ShapeShape#Encoder-1st-AdditionLayer-1/add:z:0*
T0*
_output_shapes
::��~
4Encoder-2nd-NormalizationLayer-1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
6Encoder-2nd-NormalizationLayer-1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
6Encoder-2nd-NormalizationLayer-1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
.Encoder-2nd-NormalizationLayer-1/strided_sliceStridedSlice/Encoder-2nd-NormalizationLayer-1/Shape:output:0=Encoder-2nd-NormalizationLayer-1/strided_slice/stack:output:0?Encoder-2nd-NormalizationLayer-1/strided_slice/stack_1:output:0?Encoder-2nd-NormalizationLayer-1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskp
&Encoder-2nd-NormalizationLayer-1/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
%Encoder-2nd-NormalizationLayer-1/ProdProd7Encoder-2nd-NormalizationLayer-1/strided_slice:output:0/Encoder-2nd-NormalizationLayer-1/Const:output:0*
T0*
_output_shapes
: �
6Encoder-2nd-NormalizationLayer-1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:�
8Encoder-2nd-NormalizationLayer-1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: �
8Encoder-2nd-NormalizationLayer-1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
0Encoder-2nd-NormalizationLayer-1/strided_slice_1StridedSlice/Encoder-2nd-NormalizationLayer-1/Shape:output:0?Encoder-2nd-NormalizationLayer-1/strided_slice_1/stack:output:0AEncoder-2nd-NormalizationLayer-1/strided_slice_1/stack_1:output:0AEncoder-2nd-NormalizationLayer-1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_maskr
(Encoder-2nd-NormalizationLayer-1/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
'Encoder-2nd-NormalizationLayer-1/Prod_1Prod9Encoder-2nd-NormalizationLayer-1/strided_slice_1:output:01Encoder-2nd-NormalizationLayer-1/Const_1:output:0*
T0*
_output_shapes
: r
0Encoder-2nd-NormalizationLayer-1/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :r
0Encoder-2nd-NormalizationLayer-1/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :�
.Encoder-2nd-NormalizationLayer-1/Reshape/shapePack9Encoder-2nd-NormalizationLayer-1/Reshape/shape/0:output:0.Encoder-2nd-NormalizationLayer-1/Prod:output:00Encoder-2nd-NormalizationLayer-1/Prod_1:output:09Encoder-2nd-NormalizationLayer-1/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:�
(Encoder-2nd-NormalizationLayer-1/ReshapeReshape#Encoder-1st-AdditionLayer-1/add:z:07Encoder-2nd-NormalizationLayer-1/Reshape/shape:output:0*
T0*8
_output_shapes&
$:"�������������������
,Encoder-2nd-NormalizationLayer-1/ones/packedPack.Encoder-2nd-NormalizationLayer-1/Prod:output:0*
N*
T0*
_output_shapes
:p
+Encoder-2nd-NormalizationLayer-1/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
%Encoder-2nd-NormalizationLayer-1/onesFill5Encoder-2nd-NormalizationLayer-1/ones/packed:output:04Encoder-2nd-NormalizationLayer-1/ones/Const:output:0*
T0*#
_output_shapes
:����������
-Encoder-2nd-NormalizationLayer-1/zeros/packedPack.Encoder-2nd-NormalizationLayer-1/Prod:output:0*
N*
T0*
_output_shapes
:q
,Encoder-2nd-NormalizationLayer-1/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
&Encoder-2nd-NormalizationLayer-1/zerosFill6Encoder-2nd-NormalizationLayer-1/zeros/packed:output:05Encoder-2nd-NormalizationLayer-1/zeros/Const:output:0*
T0*#
_output_shapes
:���������k
(Encoder-2nd-NormalizationLayer-1/Const_2Const*
_output_shapes
: *
dtype0*
valueB k
(Encoder-2nd-NormalizationLayer-1/Const_3Const*
_output_shapes
: *
dtype0*
valueB �
1Encoder-2nd-NormalizationLayer-1/FusedBatchNormV3FusedBatchNormV31Encoder-2nd-NormalizationLayer-1/Reshape:output:0.Encoder-2nd-NormalizationLayer-1/ones:output:0/Encoder-2nd-NormalizationLayer-1/zeros:output:01Encoder-2nd-NormalizationLayer-1/Const_2:output:01Encoder-2nd-NormalizationLayer-1/Const_3:output:0*
T0*
U0*x
_output_shapesf
d:"������������������:���������:���������:���������:���������:*
data_formatNCHW*
epsilon%o�:�
*Encoder-2nd-NormalizationLayer-1/Reshape_1Reshape5Encoder-2nd-NormalizationLayer-1/FusedBatchNormV3:y:0/Encoder-2nd-NormalizationLayer-1/Shape:output:0*
T0*+
_output_shapes
:���������P	�
3Encoder-2nd-NormalizationLayer-1/mul/ReadVariableOpReadVariableOp<encoder_2nd_normalizationlayer_1_mul_readvariableop_resource*
_output_shapes
:	*
dtype0�
$Encoder-2nd-NormalizationLayer-1/mulMul3Encoder-2nd-NormalizationLayer-1/Reshape_1:output:0;Encoder-2nd-NormalizationLayer-1/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������P	�
3Encoder-2nd-NormalizationLayer-1/add/ReadVariableOpReadVariableOp<encoder_2nd_normalizationlayer_1_add_readvariableop_resource*
_output_shapes
:	*
dtype0�
$Encoder-2nd-NormalizationLayer-1/addAddV2(Encoder-2nd-NormalizationLayer-1/mul:z:0;Encoder-2nd-NormalizationLayer-1/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������P	�
5Encoder-FeedForwardLayer_1_1/Tensordot/ReadVariableOpReadVariableOp>encoder_feedforwardlayer_1_1_tensordot_readvariableop_resource*
_output_shapes

:		*
dtype0u
+Encoder-FeedForwardLayer_1_1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:|
+Encoder-FeedForwardLayer_1_1/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       �
,Encoder-FeedForwardLayer_1_1/Tensordot/ShapeShape(Encoder-2nd-NormalizationLayer-1/add:z:0*
T0*
_output_shapes
::��v
4Encoder-FeedForwardLayer_1_1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
/Encoder-FeedForwardLayer_1_1/Tensordot/GatherV2GatherV25Encoder-FeedForwardLayer_1_1/Tensordot/Shape:output:04Encoder-FeedForwardLayer_1_1/Tensordot/free:output:0=Encoder-FeedForwardLayer_1_1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:x
6Encoder-FeedForwardLayer_1_1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
1Encoder-FeedForwardLayer_1_1/Tensordot/GatherV2_1GatherV25Encoder-FeedForwardLayer_1_1/Tensordot/Shape:output:04Encoder-FeedForwardLayer_1_1/Tensordot/axes:output:0?Encoder-FeedForwardLayer_1_1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:v
,Encoder-FeedForwardLayer_1_1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
+Encoder-FeedForwardLayer_1_1/Tensordot/ProdProd8Encoder-FeedForwardLayer_1_1/Tensordot/GatherV2:output:05Encoder-FeedForwardLayer_1_1/Tensordot/Const:output:0*
T0*
_output_shapes
: x
.Encoder-FeedForwardLayer_1_1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
-Encoder-FeedForwardLayer_1_1/Tensordot/Prod_1Prod:Encoder-FeedForwardLayer_1_1/Tensordot/GatherV2_1:output:07Encoder-FeedForwardLayer_1_1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: t
2Encoder-FeedForwardLayer_1_1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
-Encoder-FeedForwardLayer_1_1/Tensordot/concatConcatV24Encoder-FeedForwardLayer_1_1/Tensordot/free:output:04Encoder-FeedForwardLayer_1_1/Tensordot/axes:output:0;Encoder-FeedForwardLayer_1_1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
,Encoder-FeedForwardLayer_1_1/Tensordot/stackPack4Encoder-FeedForwardLayer_1_1/Tensordot/Prod:output:06Encoder-FeedForwardLayer_1_1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
0Encoder-FeedForwardLayer_1_1/Tensordot/transpose	Transpose(Encoder-2nd-NormalizationLayer-1/add:z:06Encoder-FeedForwardLayer_1_1/Tensordot/concat:output:0*
T0*+
_output_shapes
:���������P	�
.Encoder-FeedForwardLayer_1_1/Tensordot/ReshapeReshape4Encoder-FeedForwardLayer_1_1/Tensordot/transpose:y:05Encoder-FeedForwardLayer_1_1/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
-Encoder-FeedForwardLayer_1_1/Tensordot/MatMulMatMul7Encoder-FeedForwardLayer_1_1/Tensordot/Reshape:output:0=Encoder-FeedForwardLayer_1_1/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������	x
.Encoder-FeedForwardLayer_1_1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:	v
4Encoder-FeedForwardLayer_1_1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
/Encoder-FeedForwardLayer_1_1/Tensordot/concat_1ConcatV28Encoder-FeedForwardLayer_1_1/Tensordot/GatherV2:output:07Encoder-FeedForwardLayer_1_1/Tensordot/Const_2:output:0=Encoder-FeedForwardLayer_1_1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
&Encoder-FeedForwardLayer_1_1/TensordotReshape7Encoder-FeedForwardLayer_1_1/Tensordot/MatMul:product:08Encoder-FeedForwardLayer_1_1/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������P	�
3Encoder-FeedForwardLayer_1_1/BiasAdd/ReadVariableOpReadVariableOp<encoder_feedforwardlayer_1_1_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype0�
$Encoder-FeedForwardLayer_1_1/BiasAddBiasAdd/Encoder-FeedForwardLayer_1_1/Tensordot:output:0;Encoder-FeedForwardLayer_1_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������P	l
'Encoder-FeedForwardLayer_1_1/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
%Encoder-FeedForwardLayer_1_1/Gelu/mulMul0Encoder-FeedForwardLayer_1_1/Gelu/mul/x:output:0-Encoder-FeedForwardLayer_1_1/BiasAdd:output:0*
T0*+
_output_shapes
:���������P	m
(Encoder-FeedForwardLayer_1_1/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *��?�
)Encoder-FeedForwardLayer_1_1/Gelu/truedivRealDiv-Encoder-FeedForwardLayer_1_1/BiasAdd:output:01Encoder-FeedForwardLayer_1_1/Gelu/Cast/x:output:0*
T0*+
_output_shapes
:���������P	�
%Encoder-FeedForwardLayer_1_1/Gelu/ErfErf-Encoder-FeedForwardLayer_1_1/Gelu/truediv:z:0*
T0*+
_output_shapes
:���������P	l
'Encoder-FeedForwardLayer_1_1/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
%Encoder-FeedForwardLayer_1_1/Gelu/addAddV20Encoder-FeedForwardLayer_1_1/Gelu/add/x:output:0)Encoder-FeedForwardLayer_1_1/Gelu/Erf:y:0*
T0*+
_output_shapes
:���������P	�
'Encoder-FeedForwardLayer_1_1/Gelu/mul_1Mul)Encoder-FeedForwardLayer_1_1/Gelu/mul:z:0)Encoder-FeedForwardLayer_1_1/Gelu/add:z:0*
T0*+
_output_shapes
:���������P	�
5Encoder-FeedForwardLayer_2_1/Tensordot/ReadVariableOpReadVariableOp>encoder_feedforwardlayer_2_1_tensordot_readvariableop_resource*
_output_shapes

:		*
dtype0u
+Encoder-FeedForwardLayer_2_1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:|
+Encoder-FeedForwardLayer_2_1/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       �
,Encoder-FeedForwardLayer_2_1/Tensordot/ShapeShape+Encoder-FeedForwardLayer_1_1/Gelu/mul_1:z:0*
T0*
_output_shapes
::��v
4Encoder-FeedForwardLayer_2_1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
/Encoder-FeedForwardLayer_2_1/Tensordot/GatherV2GatherV25Encoder-FeedForwardLayer_2_1/Tensordot/Shape:output:04Encoder-FeedForwardLayer_2_1/Tensordot/free:output:0=Encoder-FeedForwardLayer_2_1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:x
6Encoder-FeedForwardLayer_2_1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
1Encoder-FeedForwardLayer_2_1/Tensordot/GatherV2_1GatherV25Encoder-FeedForwardLayer_2_1/Tensordot/Shape:output:04Encoder-FeedForwardLayer_2_1/Tensordot/axes:output:0?Encoder-FeedForwardLayer_2_1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:v
,Encoder-FeedForwardLayer_2_1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
+Encoder-FeedForwardLayer_2_1/Tensordot/ProdProd8Encoder-FeedForwardLayer_2_1/Tensordot/GatherV2:output:05Encoder-FeedForwardLayer_2_1/Tensordot/Const:output:0*
T0*
_output_shapes
: x
.Encoder-FeedForwardLayer_2_1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
-Encoder-FeedForwardLayer_2_1/Tensordot/Prod_1Prod:Encoder-FeedForwardLayer_2_1/Tensordot/GatherV2_1:output:07Encoder-FeedForwardLayer_2_1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: t
2Encoder-FeedForwardLayer_2_1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
-Encoder-FeedForwardLayer_2_1/Tensordot/concatConcatV24Encoder-FeedForwardLayer_2_1/Tensordot/free:output:04Encoder-FeedForwardLayer_2_1/Tensordot/axes:output:0;Encoder-FeedForwardLayer_2_1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
,Encoder-FeedForwardLayer_2_1/Tensordot/stackPack4Encoder-FeedForwardLayer_2_1/Tensordot/Prod:output:06Encoder-FeedForwardLayer_2_1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
0Encoder-FeedForwardLayer_2_1/Tensordot/transpose	Transpose+Encoder-FeedForwardLayer_1_1/Gelu/mul_1:z:06Encoder-FeedForwardLayer_2_1/Tensordot/concat:output:0*
T0*+
_output_shapes
:���������P	�
.Encoder-FeedForwardLayer_2_1/Tensordot/ReshapeReshape4Encoder-FeedForwardLayer_2_1/Tensordot/transpose:y:05Encoder-FeedForwardLayer_2_1/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
-Encoder-FeedForwardLayer_2_1/Tensordot/MatMulMatMul7Encoder-FeedForwardLayer_2_1/Tensordot/Reshape:output:0=Encoder-FeedForwardLayer_2_1/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������	x
.Encoder-FeedForwardLayer_2_1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:	v
4Encoder-FeedForwardLayer_2_1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
/Encoder-FeedForwardLayer_2_1/Tensordot/concat_1ConcatV28Encoder-FeedForwardLayer_2_1/Tensordot/GatherV2:output:07Encoder-FeedForwardLayer_2_1/Tensordot/Const_2:output:0=Encoder-FeedForwardLayer_2_1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
&Encoder-FeedForwardLayer_2_1/TensordotReshape7Encoder-FeedForwardLayer_2_1/Tensordot/MatMul:product:08Encoder-FeedForwardLayer_2_1/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������P	�
3Encoder-FeedForwardLayer_2_1/BiasAdd/ReadVariableOpReadVariableOp<encoder_feedforwardlayer_2_1_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype0�
$Encoder-FeedForwardLayer_2_1/BiasAddBiasAdd/Encoder-FeedForwardLayer_2_1/Tensordot:output:0;Encoder-FeedForwardLayer_2_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������P	�
dropout_3/IdentityIdentity-Encoder-FeedForwardLayer_2_1/BiasAdd:output:0*
T0*+
_output_shapes
:���������P	�
Encoder-2nd-AdditionLayer-1/addAddV2#Encoder-1st-AdditionLayer-1/add:z:0dropout_3/Identity:output:0*
T0*+
_output_shapes
:���������P	v
IdentityIdentity#Encoder-2nd-AdditionLayer-1/add:z:0^NoOp*
T0*+
_output_shapes
:���������P	�

Identity_1Identity6Encoder-SelfAttentionLayer-1/softmax/Softmax:softmax:0^NoOp*
T0*/
_output_shapes
:���������PP�
NoOpNoOp4^Encoder-1st-NormalizationLayer-1/add/ReadVariableOp4^Encoder-1st-NormalizationLayer-1/mul/ReadVariableOp4^Encoder-2nd-NormalizationLayer-1/add/ReadVariableOp4^Encoder-2nd-NormalizationLayer-1/mul/ReadVariableOp4^Encoder-FeedForwardLayer_1_1/BiasAdd/ReadVariableOp6^Encoder-FeedForwardLayer_1_1/Tensordot/ReadVariableOp4^Encoder-FeedForwardLayer_2_1/BiasAdd/ReadVariableOp6^Encoder-FeedForwardLayer_2_1/Tensordot/ReadVariableOpA^Encoder-SelfAttentionLayer-1/attention_output/add/ReadVariableOpK^Encoder-SelfAttentionLayer-1/attention_output/einsum/Einsum/ReadVariableOp4^Encoder-SelfAttentionLayer-1/key/add/ReadVariableOp>^Encoder-SelfAttentionLayer-1/key/einsum/Einsum/ReadVariableOp6^Encoder-SelfAttentionLayer-1/query/add/ReadVariableOp@^Encoder-SelfAttentionLayer-1/query/einsum/Einsum/ReadVariableOp6^Encoder-SelfAttentionLayer-1/value/add/ReadVariableOp@^Encoder-SelfAttentionLayer-1/value/einsum/Einsum/ReadVariableOp*
_output_shapes
 "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:���������P	: : : : : : : : : : : : : : : : 2j
3Encoder-1st-NormalizationLayer-1/add/ReadVariableOp3Encoder-1st-NormalizationLayer-1/add/ReadVariableOp2j
3Encoder-1st-NormalizationLayer-1/mul/ReadVariableOp3Encoder-1st-NormalizationLayer-1/mul/ReadVariableOp2j
3Encoder-2nd-NormalizationLayer-1/add/ReadVariableOp3Encoder-2nd-NormalizationLayer-1/add/ReadVariableOp2j
3Encoder-2nd-NormalizationLayer-1/mul/ReadVariableOp3Encoder-2nd-NormalizationLayer-1/mul/ReadVariableOp2j
3Encoder-FeedForwardLayer_1_1/BiasAdd/ReadVariableOp3Encoder-FeedForwardLayer_1_1/BiasAdd/ReadVariableOp2n
5Encoder-FeedForwardLayer_1_1/Tensordot/ReadVariableOp5Encoder-FeedForwardLayer_1_1/Tensordot/ReadVariableOp2j
3Encoder-FeedForwardLayer_2_1/BiasAdd/ReadVariableOp3Encoder-FeedForwardLayer_2_1/BiasAdd/ReadVariableOp2n
5Encoder-FeedForwardLayer_2_1/Tensordot/ReadVariableOp5Encoder-FeedForwardLayer_2_1/Tensordot/ReadVariableOp2�
@Encoder-SelfAttentionLayer-1/attention_output/add/ReadVariableOp@Encoder-SelfAttentionLayer-1/attention_output/add/ReadVariableOp2�
JEncoder-SelfAttentionLayer-1/attention_output/einsum/Einsum/ReadVariableOpJEncoder-SelfAttentionLayer-1/attention_output/einsum/Einsum/ReadVariableOp2j
3Encoder-SelfAttentionLayer-1/key/add/ReadVariableOp3Encoder-SelfAttentionLayer-1/key/add/ReadVariableOp2~
=Encoder-SelfAttentionLayer-1/key/einsum/Einsum/ReadVariableOp=Encoder-SelfAttentionLayer-1/key/einsum/Einsum/ReadVariableOp2n
5Encoder-SelfAttentionLayer-1/query/add/ReadVariableOp5Encoder-SelfAttentionLayer-1/query/add/ReadVariableOp2�
?Encoder-SelfAttentionLayer-1/query/einsum/Einsum/ReadVariableOp?Encoder-SelfAttentionLayer-1/query/einsum/Einsum/ReadVariableOp2n
5Encoder-SelfAttentionLayer-1/value/add/ReadVariableOp5Encoder-SelfAttentionLayer-1/value/add/ReadVariableOp2�
?Encoder-SelfAttentionLayer-1/value/einsum/Einsum/ReadVariableOp?Encoder-SelfAttentionLayer-1/value/einsum/Einsum/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:(
$
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:S O
+
_output_shapes
:���������P	
 
_user_specified_nameinputs
�
�
6__inference_transformer_encoder_4_layer_call_fn_232407

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:
	unknown_3:	
	unknown_4:
	unknown_5:	
	unknown_6:
	unknown_7:	
	unknown_8:	
	unknown_9:	

unknown_10:	

unknown_11:		

unknown_12:	

unknown_13:		

unknown_14:	
identity

identity_1��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *F
_output_shapes4
2:���������P	:���������PP*2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *Z
fURS
Q__inference_transformer_encoder_4_layer_call_and_return_conditional_losses_230943s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������P	y

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*/
_output_shapes
:���������PP<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:���������P	: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name232401:&"
 
_user_specified_name232399:&"
 
_user_specified_name232397:&"
 
_user_specified_name232395:&"
 
_user_specified_name232393:&"
 
_user_specified_name232391:&
"
 
_user_specified_name232389:&	"
 
_user_specified_name232387:&"
 
_user_specified_name232385:&"
 
_user_specified_name232383:&"
 
_user_specified_name232381:&"
 
_user_specified_name232379:&"
 
_user_specified_name232377:&"
 
_user_specified_name232375:&"
 
_user_specified_name232373:&"
 
_user_specified_name232371:S O
+
_output_shapes
:���������P	
 
_user_specified_nameinputs
�
^
B__inference_Output_layer_call_and_return_conditional_losses_233392

inputs
identity=
SqueezeSqueezeinputs*
T0*
_output_shapes
:I
IdentityIdentitySqueeze:output:0*
T0*
_output_shapes
:"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�

�
Q__inference_PredictionImprovement_layer_call_and_return_conditional_losses_230543

inputs0
matmul_readvariableop_resource:
-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:���������Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:���������S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
'
_output_shapes
:���������

 
_user_specified_nameinputs
�
�
/__inference_FinalLayerNorm_layer_call_fn_233218

inputs
unknown:	
	unknown_0:	
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������P	*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_FinalLayerNorm_layer_call_and_return_conditional_losses_230477s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������P	<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������P	: : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name233214:&"
 
_user_specified_name233212:S O
+
_output_shapes
:���������P	
 
_user_specified_nameinputs
��
�
Q__inference_transformer_encoder_3_layer_call_and_return_conditional_losses_230735

inputsJ
<encoder_1st_normalizationlayer_1_mul_readvariableop_resource:	J
<encoder_1st_normalizationlayer_1_add_readvariableop_resource:	^
Hencoder_selfattentionlayer_1_query_einsum_einsum_readvariableop_resource:	P
>encoder_selfattentionlayer_1_query_add_readvariableop_resource:\
Fencoder_selfattentionlayer_1_key_einsum_einsum_readvariableop_resource:	N
<encoder_selfattentionlayer_1_key_add_readvariableop_resource:^
Hencoder_selfattentionlayer_1_value_einsum_einsum_readvariableop_resource:	P
>encoder_selfattentionlayer_1_value_add_readvariableop_resource:i
Sencoder_selfattentionlayer_1_attention_output_einsum_einsum_readvariableop_resource:	W
Iencoder_selfattentionlayer_1_attention_output_add_readvariableop_resource:	J
<encoder_2nd_normalizationlayer_1_mul_readvariableop_resource:	J
<encoder_2nd_normalizationlayer_1_add_readvariableop_resource:	P
>encoder_feedforwardlayer_1_1_tensordot_readvariableop_resource:		J
<encoder_feedforwardlayer_1_1_biasadd_readvariableop_resource:	P
>encoder_feedforwardlayer_2_1_tensordot_readvariableop_resource:		J
<encoder_feedforwardlayer_2_1_biasadd_readvariableop_resource:	
identity

identity_1��3Encoder-1st-NormalizationLayer-1/add/ReadVariableOp�3Encoder-1st-NormalizationLayer-1/mul/ReadVariableOp�3Encoder-2nd-NormalizationLayer-1/add/ReadVariableOp�3Encoder-2nd-NormalizationLayer-1/mul/ReadVariableOp�3Encoder-FeedForwardLayer_1_1/BiasAdd/ReadVariableOp�5Encoder-FeedForwardLayer_1_1/Tensordot/ReadVariableOp�3Encoder-FeedForwardLayer_2_1/BiasAdd/ReadVariableOp�5Encoder-FeedForwardLayer_2_1/Tensordot/ReadVariableOp�@Encoder-SelfAttentionLayer-1/attention_output/add/ReadVariableOp�JEncoder-SelfAttentionLayer-1/attention_output/einsum/Einsum/ReadVariableOp�3Encoder-SelfAttentionLayer-1/key/add/ReadVariableOp�=Encoder-SelfAttentionLayer-1/key/einsum/Einsum/ReadVariableOp�5Encoder-SelfAttentionLayer-1/query/add/ReadVariableOp�?Encoder-SelfAttentionLayer-1/query/einsum/Einsum/ReadVariableOp�5Encoder-SelfAttentionLayer-1/value/add/ReadVariableOp�?Encoder-SelfAttentionLayer-1/value/einsum/Einsum/ReadVariableOpj
&Encoder-1st-NormalizationLayer-1/ShapeShapeinputs*
T0*
_output_shapes
::��~
4Encoder-1st-NormalizationLayer-1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
6Encoder-1st-NormalizationLayer-1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
6Encoder-1st-NormalizationLayer-1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
.Encoder-1st-NormalizationLayer-1/strided_sliceStridedSlice/Encoder-1st-NormalizationLayer-1/Shape:output:0=Encoder-1st-NormalizationLayer-1/strided_slice/stack:output:0?Encoder-1st-NormalizationLayer-1/strided_slice/stack_1:output:0?Encoder-1st-NormalizationLayer-1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskp
&Encoder-1st-NormalizationLayer-1/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
%Encoder-1st-NormalizationLayer-1/ProdProd7Encoder-1st-NormalizationLayer-1/strided_slice:output:0/Encoder-1st-NormalizationLayer-1/Const:output:0*
T0*
_output_shapes
: �
6Encoder-1st-NormalizationLayer-1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:�
8Encoder-1st-NormalizationLayer-1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: �
8Encoder-1st-NormalizationLayer-1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
0Encoder-1st-NormalizationLayer-1/strided_slice_1StridedSlice/Encoder-1st-NormalizationLayer-1/Shape:output:0?Encoder-1st-NormalizationLayer-1/strided_slice_1/stack:output:0AEncoder-1st-NormalizationLayer-1/strided_slice_1/stack_1:output:0AEncoder-1st-NormalizationLayer-1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_maskr
(Encoder-1st-NormalizationLayer-1/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
'Encoder-1st-NormalizationLayer-1/Prod_1Prod9Encoder-1st-NormalizationLayer-1/strided_slice_1:output:01Encoder-1st-NormalizationLayer-1/Const_1:output:0*
T0*
_output_shapes
: r
0Encoder-1st-NormalizationLayer-1/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :r
0Encoder-1st-NormalizationLayer-1/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :�
.Encoder-1st-NormalizationLayer-1/Reshape/shapePack9Encoder-1st-NormalizationLayer-1/Reshape/shape/0:output:0.Encoder-1st-NormalizationLayer-1/Prod:output:00Encoder-1st-NormalizationLayer-1/Prod_1:output:09Encoder-1st-NormalizationLayer-1/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:�
(Encoder-1st-NormalizationLayer-1/ReshapeReshapeinputs7Encoder-1st-NormalizationLayer-1/Reshape/shape:output:0*
T0*8
_output_shapes&
$:"�������������������
,Encoder-1st-NormalizationLayer-1/ones/packedPack.Encoder-1st-NormalizationLayer-1/Prod:output:0*
N*
T0*
_output_shapes
:p
+Encoder-1st-NormalizationLayer-1/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
%Encoder-1st-NormalizationLayer-1/onesFill5Encoder-1st-NormalizationLayer-1/ones/packed:output:04Encoder-1st-NormalizationLayer-1/ones/Const:output:0*
T0*#
_output_shapes
:����������
-Encoder-1st-NormalizationLayer-1/zeros/packedPack.Encoder-1st-NormalizationLayer-1/Prod:output:0*
N*
T0*
_output_shapes
:q
,Encoder-1st-NormalizationLayer-1/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
&Encoder-1st-NormalizationLayer-1/zerosFill6Encoder-1st-NormalizationLayer-1/zeros/packed:output:05Encoder-1st-NormalizationLayer-1/zeros/Const:output:0*
T0*#
_output_shapes
:���������k
(Encoder-1st-NormalizationLayer-1/Const_2Const*
_output_shapes
: *
dtype0*
valueB k
(Encoder-1st-NormalizationLayer-1/Const_3Const*
_output_shapes
: *
dtype0*
valueB �
1Encoder-1st-NormalizationLayer-1/FusedBatchNormV3FusedBatchNormV31Encoder-1st-NormalizationLayer-1/Reshape:output:0.Encoder-1st-NormalizationLayer-1/ones:output:0/Encoder-1st-NormalizationLayer-1/zeros:output:01Encoder-1st-NormalizationLayer-1/Const_2:output:01Encoder-1st-NormalizationLayer-1/Const_3:output:0*
T0*
U0*x
_output_shapesf
d:"������������������:���������:���������:���������:���������:*
data_formatNCHW*
epsilon%o�:�
*Encoder-1st-NormalizationLayer-1/Reshape_1Reshape5Encoder-1st-NormalizationLayer-1/FusedBatchNormV3:y:0/Encoder-1st-NormalizationLayer-1/Shape:output:0*
T0*+
_output_shapes
:���������P	�
3Encoder-1st-NormalizationLayer-1/mul/ReadVariableOpReadVariableOp<encoder_1st_normalizationlayer_1_mul_readvariableop_resource*
_output_shapes
:	*
dtype0�
$Encoder-1st-NormalizationLayer-1/mulMul3Encoder-1st-NormalizationLayer-1/Reshape_1:output:0;Encoder-1st-NormalizationLayer-1/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������P	�
3Encoder-1st-NormalizationLayer-1/add/ReadVariableOpReadVariableOp<encoder_1st_normalizationlayer_1_add_readvariableop_resource*
_output_shapes
:	*
dtype0�
$Encoder-1st-NormalizationLayer-1/addAddV2(Encoder-1st-NormalizationLayer-1/mul:z:0;Encoder-1st-NormalizationLayer-1/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������P	�
?Encoder-SelfAttentionLayer-1/query/einsum/Einsum/ReadVariableOpReadVariableOpHencoder_selfattentionlayer_1_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:	*
dtype0�
0Encoder-SelfAttentionLayer-1/query/einsum/EinsumEinsum(Encoder-1st-NormalizationLayer-1/add:z:0GEncoder-SelfAttentionLayer-1/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������P*
equationabc,cde->abde�
5Encoder-SelfAttentionLayer-1/query/add/ReadVariableOpReadVariableOp>encoder_selfattentionlayer_1_query_add_readvariableop_resource*
_output_shapes

:*
dtype0�
&Encoder-SelfAttentionLayer-1/query/addAddV29Encoder-SelfAttentionLayer-1/query/einsum/Einsum:output:0=Encoder-SelfAttentionLayer-1/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������P�
=Encoder-SelfAttentionLayer-1/key/einsum/Einsum/ReadVariableOpReadVariableOpFencoder_selfattentionlayer_1_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:	*
dtype0�
.Encoder-SelfAttentionLayer-1/key/einsum/EinsumEinsum(Encoder-1st-NormalizationLayer-1/add:z:0EEncoder-SelfAttentionLayer-1/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������P*
equationabc,cde->abde�
3Encoder-SelfAttentionLayer-1/key/add/ReadVariableOpReadVariableOp<encoder_selfattentionlayer_1_key_add_readvariableop_resource*
_output_shapes

:*
dtype0�
$Encoder-SelfAttentionLayer-1/key/addAddV27Encoder-SelfAttentionLayer-1/key/einsum/Einsum:output:0;Encoder-SelfAttentionLayer-1/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������P�
?Encoder-SelfAttentionLayer-1/value/einsum/Einsum/ReadVariableOpReadVariableOpHencoder_selfattentionlayer_1_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:	*
dtype0�
0Encoder-SelfAttentionLayer-1/value/einsum/EinsumEinsum(Encoder-1st-NormalizationLayer-1/add:z:0GEncoder-SelfAttentionLayer-1/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������P*
equationabc,cde->abde�
5Encoder-SelfAttentionLayer-1/value/add/ReadVariableOpReadVariableOp>encoder_selfattentionlayer_1_value_add_readvariableop_resource*
_output_shapes

:*
dtype0�
&Encoder-SelfAttentionLayer-1/value/addAddV29Encoder-SelfAttentionLayer-1/value/einsum/Einsum:output:0=Encoder-SelfAttentionLayer-1/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������Pg
"Encoder-SelfAttentionLayer-1/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *:�?�
 Encoder-SelfAttentionLayer-1/MulMul*Encoder-SelfAttentionLayer-1/query/add:z:0+Encoder-SelfAttentionLayer-1/Mul/y:output:0*
T0*/
_output_shapes
:���������P�
*Encoder-SelfAttentionLayer-1/einsum/EinsumEinsum(Encoder-SelfAttentionLayer-1/key/add:z:0$Encoder-SelfAttentionLayer-1/Mul:z:0*
N*
T0*/
_output_shapes
:���������PP*
equationaecd,abcd->acbe�
,Encoder-SelfAttentionLayer-1/softmax/SoftmaxSoftmax3Encoder-SelfAttentionLayer-1/einsum/Einsum:output:0*
T0*/
_output_shapes
:���������PP�
-Encoder-SelfAttentionLayer-1/dropout/IdentityIdentity6Encoder-SelfAttentionLayer-1/softmax/Softmax:softmax:0*
T0*/
_output_shapes
:���������PP�
,Encoder-SelfAttentionLayer-1/einsum_1/EinsumEinsum6Encoder-SelfAttentionLayer-1/dropout/Identity:output:0*Encoder-SelfAttentionLayer-1/value/add:z:0*
N*
T0*/
_output_shapes
:���������P*
equationacbe,aecd->abcd�
JEncoder-SelfAttentionLayer-1/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpSencoder_selfattentionlayer_1_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:	*
dtype0�
;Encoder-SelfAttentionLayer-1/attention_output/einsum/EinsumEinsum5Encoder-SelfAttentionLayer-1/einsum_1/Einsum:output:0REncoder-SelfAttentionLayer-1/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:���������P	*
equationabcd,cde->abe�
@Encoder-SelfAttentionLayer-1/attention_output/add/ReadVariableOpReadVariableOpIencoder_selfattentionlayer_1_attention_output_add_readvariableop_resource*
_output_shapes
:	*
dtype0�
1Encoder-SelfAttentionLayer-1/attention_output/addAddV2DEncoder-SelfAttentionLayer-1/attention_output/einsum/Einsum:output:0HEncoder-SelfAttentionLayer-1/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������P	�
Encoder-1st-AdditionLayer-1/addAddV2inputs5Encoder-SelfAttentionLayer-1/attention_output/add:z:0*
T0*+
_output_shapes
:���������P	�
&Encoder-2nd-NormalizationLayer-1/ShapeShape#Encoder-1st-AdditionLayer-1/add:z:0*
T0*
_output_shapes
::��~
4Encoder-2nd-NormalizationLayer-1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
6Encoder-2nd-NormalizationLayer-1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
6Encoder-2nd-NormalizationLayer-1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
.Encoder-2nd-NormalizationLayer-1/strided_sliceStridedSlice/Encoder-2nd-NormalizationLayer-1/Shape:output:0=Encoder-2nd-NormalizationLayer-1/strided_slice/stack:output:0?Encoder-2nd-NormalizationLayer-1/strided_slice/stack_1:output:0?Encoder-2nd-NormalizationLayer-1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskp
&Encoder-2nd-NormalizationLayer-1/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
%Encoder-2nd-NormalizationLayer-1/ProdProd7Encoder-2nd-NormalizationLayer-1/strided_slice:output:0/Encoder-2nd-NormalizationLayer-1/Const:output:0*
T0*
_output_shapes
: �
6Encoder-2nd-NormalizationLayer-1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:�
8Encoder-2nd-NormalizationLayer-1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: �
8Encoder-2nd-NormalizationLayer-1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
0Encoder-2nd-NormalizationLayer-1/strided_slice_1StridedSlice/Encoder-2nd-NormalizationLayer-1/Shape:output:0?Encoder-2nd-NormalizationLayer-1/strided_slice_1/stack:output:0AEncoder-2nd-NormalizationLayer-1/strided_slice_1/stack_1:output:0AEncoder-2nd-NormalizationLayer-1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_maskr
(Encoder-2nd-NormalizationLayer-1/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
'Encoder-2nd-NormalizationLayer-1/Prod_1Prod9Encoder-2nd-NormalizationLayer-1/strided_slice_1:output:01Encoder-2nd-NormalizationLayer-1/Const_1:output:0*
T0*
_output_shapes
: r
0Encoder-2nd-NormalizationLayer-1/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :r
0Encoder-2nd-NormalizationLayer-1/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :�
.Encoder-2nd-NormalizationLayer-1/Reshape/shapePack9Encoder-2nd-NormalizationLayer-1/Reshape/shape/0:output:0.Encoder-2nd-NormalizationLayer-1/Prod:output:00Encoder-2nd-NormalizationLayer-1/Prod_1:output:09Encoder-2nd-NormalizationLayer-1/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:�
(Encoder-2nd-NormalizationLayer-1/ReshapeReshape#Encoder-1st-AdditionLayer-1/add:z:07Encoder-2nd-NormalizationLayer-1/Reshape/shape:output:0*
T0*8
_output_shapes&
$:"�������������������
,Encoder-2nd-NormalizationLayer-1/ones/packedPack.Encoder-2nd-NormalizationLayer-1/Prod:output:0*
N*
T0*
_output_shapes
:p
+Encoder-2nd-NormalizationLayer-1/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
%Encoder-2nd-NormalizationLayer-1/onesFill5Encoder-2nd-NormalizationLayer-1/ones/packed:output:04Encoder-2nd-NormalizationLayer-1/ones/Const:output:0*
T0*#
_output_shapes
:����������
-Encoder-2nd-NormalizationLayer-1/zeros/packedPack.Encoder-2nd-NormalizationLayer-1/Prod:output:0*
N*
T0*
_output_shapes
:q
,Encoder-2nd-NormalizationLayer-1/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
&Encoder-2nd-NormalizationLayer-1/zerosFill6Encoder-2nd-NormalizationLayer-1/zeros/packed:output:05Encoder-2nd-NormalizationLayer-1/zeros/Const:output:0*
T0*#
_output_shapes
:���������k
(Encoder-2nd-NormalizationLayer-1/Const_2Const*
_output_shapes
: *
dtype0*
valueB k
(Encoder-2nd-NormalizationLayer-1/Const_3Const*
_output_shapes
: *
dtype0*
valueB �
1Encoder-2nd-NormalizationLayer-1/FusedBatchNormV3FusedBatchNormV31Encoder-2nd-NormalizationLayer-1/Reshape:output:0.Encoder-2nd-NormalizationLayer-1/ones:output:0/Encoder-2nd-NormalizationLayer-1/zeros:output:01Encoder-2nd-NormalizationLayer-1/Const_2:output:01Encoder-2nd-NormalizationLayer-1/Const_3:output:0*
T0*
U0*x
_output_shapesf
d:"������������������:���������:���������:���������:���������:*
data_formatNCHW*
epsilon%o�:�
*Encoder-2nd-NormalizationLayer-1/Reshape_1Reshape5Encoder-2nd-NormalizationLayer-1/FusedBatchNormV3:y:0/Encoder-2nd-NormalizationLayer-1/Shape:output:0*
T0*+
_output_shapes
:���������P	�
3Encoder-2nd-NormalizationLayer-1/mul/ReadVariableOpReadVariableOp<encoder_2nd_normalizationlayer_1_mul_readvariableop_resource*
_output_shapes
:	*
dtype0�
$Encoder-2nd-NormalizationLayer-1/mulMul3Encoder-2nd-NormalizationLayer-1/Reshape_1:output:0;Encoder-2nd-NormalizationLayer-1/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������P	�
3Encoder-2nd-NormalizationLayer-1/add/ReadVariableOpReadVariableOp<encoder_2nd_normalizationlayer_1_add_readvariableop_resource*
_output_shapes
:	*
dtype0�
$Encoder-2nd-NormalizationLayer-1/addAddV2(Encoder-2nd-NormalizationLayer-1/mul:z:0;Encoder-2nd-NormalizationLayer-1/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������P	�
5Encoder-FeedForwardLayer_1_1/Tensordot/ReadVariableOpReadVariableOp>encoder_feedforwardlayer_1_1_tensordot_readvariableop_resource*
_output_shapes

:		*
dtype0u
+Encoder-FeedForwardLayer_1_1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:|
+Encoder-FeedForwardLayer_1_1/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       �
,Encoder-FeedForwardLayer_1_1/Tensordot/ShapeShape(Encoder-2nd-NormalizationLayer-1/add:z:0*
T0*
_output_shapes
::��v
4Encoder-FeedForwardLayer_1_1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
/Encoder-FeedForwardLayer_1_1/Tensordot/GatherV2GatherV25Encoder-FeedForwardLayer_1_1/Tensordot/Shape:output:04Encoder-FeedForwardLayer_1_1/Tensordot/free:output:0=Encoder-FeedForwardLayer_1_1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:x
6Encoder-FeedForwardLayer_1_1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
1Encoder-FeedForwardLayer_1_1/Tensordot/GatherV2_1GatherV25Encoder-FeedForwardLayer_1_1/Tensordot/Shape:output:04Encoder-FeedForwardLayer_1_1/Tensordot/axes:output:0?Encoder-FeedForwardLayer_1_1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:v
,Encoder-FeedForwardLayer_1_1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
+Encoder-FeedForwardLayer_1_1/Tensordot/ProdProd8Encoder-FeedForwardLayer_1_1/Tensordot/GatherV2:output:05Encoder-FeedForwardLayer_1_1/Tensordot/Const:output:0*
T0*
_output_shapes
: x
.Encoder-FeedForwardLayer_1_1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
-Encoder-FeedForwardLayer_1_1/Tensordot/Prod_1Prod:Encoder-FeedForwardLayer_1_1/Tensordot/GatherV2_1:output:07Encoder-FeedForwardLayer_1_1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: t
2Encoder-FeedForwardLayer_1_1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
-Encoder-FeedForwardLayer_1_1/Tensordot/concatConcatV24Encoder-FeedForwardLayer_1_1/Tensordot/free:output:04Encoder-FeedForwardLayer_1_1/Tensordot/axes:output:0;Encoder-FeedForwardLayer_1_1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
,Encoder-FeedForwardLayer_1_1/Tensordot/stackPack4Encoder-FeedForwardLayer_1_1/Tensordot/Prod:output:06Encoder-FeedForwardLayer_1_1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
0Encoder-FeedForwardLayer_1_1/Tensordot/transpose	Transpose(Encoder-2nd-NormalizationLayer-1/add:z:06Encoder-FeedForwardLayer_1_1/Tensordot/concat:output:0*
T0*+
_output_shapes
:���������P	�
.Encoder-FeedForwardLayer_1_1/Tensordot/ReshapeReshape4Encoder-FeedForwardLayer_1_1/Tensordot/transpose:y:05Encoder-FeedForwardLayer_1_1/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
-Encoder-FeedForwardLayer_1_1/Tensordot/MatMulMatMul7Encoder-FeedForwardLayer_1_1/Tensordot/Reshape:output:0=Encoder-FeedForwardLayer_1_1/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������	x
.Encoder-FeedForwardLayer_1_1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:	v
4Encoder-FeedForwardLayer_1_1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
/Encoder-FeedForwardLayer_1_1/Tensordot/concat_1ConcatV28Encoder-FeedForwardLayer_1_1/Tensordot/GatherV2:output:07Encoder-FeedForwardLayer_1_1/Tensordot/Const_2:output:0=Encoder-FeedForwardLayer_1_1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
&Encoder-FeedForwardLayer_1_1/TensordotReshape7Encoder-FeedForwardLayer_1_1/Tensordot/MatMul:product:08Encoder-FeedForwardLayer_1_1/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������P	�
3Encoder-FeedForwardLayer_1_1/BiasAdd/ReadVariableOpReadVariableOp<encoder_feedforwardlayer_1_1_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype0�
$Encoder-FeedForwardLayer_1_1/BiasAddBiasAdd/Encoder-FeedForwardLayer_1_1/Tensordot:output:0;Encoder-FeedForwardLayer_1_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������P	l
'Encoder-FeedForwardLayer_1_1/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
%Encoder-FeedForwardLayer_1_1/Gelu/mulMul0Encoder-FeedForwardLayer_1_1/Gelu/mul/x:output:0-Encoder-FeedForwardLayer_1_1/BiasAdd:output:0*
T0*+
_output_shapes
:���������P	m
(Encoder-FeedForwardLayer_1_1/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *��?�
)Encoder-FeedForwardLayer_1_1/Gelu/truedivRealDiv-Encoder-FeedForwardLayer_1_1/BiasAdd:output:01Encoder-FeedForwardLayer_1_1/Gelu/Cast/x:output:0*
T0*+
_output_shapes
:���������P	�
%Encoder-FeedForwardLayer_1_1/Gelu/ErfErf-Encoder-FeedForwardLayer_1_1/Gelu/truediv:z:0*
T0*+
_output_shapes
:���������P	l
'Encoder-FeedForwardLayer_1_1/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
%Encoder-FeedForwardLayer_1_1/Gelu/addAddV20Encoder-FeedForwardLayer_1_1/Gelu/add/x:output:0)Encoder-FeedForwardLayer_1_1/Gelu/Erf:y:0*
T0*+
_output_shapes
:���������P	�
'Encoder-FeedForwardLayer_1_1/Gelu/mul_1Mul)Encoder-FeedForwardLayer_1_1/Gelu/mul:z:0)Encoder-FeedForwardLayer_1_1/Gelu/add:z:0*
T0*+
_output_shapes
:���������P	�
5Encoder-FeedForwardLayer_2_1/Tensordot/ReadVariableOpReadVariableOp>encoder_feedforwardlayer_2_1_tensordot_readvariableop_resource*
_output_shapes

:		*
dtype0u
+Encoder-FeedForwardLayer_2_1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:|
+Encoder-FeedForwardLayer_2_1/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       �
,Encoder-FeedForwardLayer_2_1/Tensordot/ShapeShape+Encoder-FeedForwardLayer_1_1/Gelu/mul_1:z:0*
T0*
_output_shapes
::��v
4Encoder-FeedForwardLayer_2_1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
/Encoder-FeedForwardLayer_2_1/Tensordot/GatherV2GatherV25Encoder-FeedForwardLayer_2_1/Tensordot/Shape:output:04Encoder-FeedForwardLayer_2_1/Tensordot/free:output:0=Encoder-FeedForwardLayer_2_1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:x
6Encoder-FeedForwardLayer_2_1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
1Encoder-FeedForwardLayer_2_1/Tensordot/GatherV2_1GatherV25Encoder-FeedForwardLayer_2_1/Tensordot/Shape:output:04Encoder-FeedForwardLayer_2_1/Tensordot/axes:output:0?Encoder-FeedForwardLayer_2_1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:v
,Encoder-FeedForwardLayer_2_1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
+Encoder-FeedForwardLayer_2_1/Tensordot/ProdProd8Encoder-FeedForwardLayer_2_1/Tensordot/GatherV2:output:05Encoder-FeedForwardLayer_2_1/Tensordot/Const:output:0*
T0*
_output_shapes
: x
.Encoder-FeedForwardLayer_2_1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
-Encoder-FeedForwardLayer_2_1/Tensordot/Prod_1Prod:Encoder-FeedForwardLayer_2_1/Tensordot/GatherV2_1:output:07Encoder-FeedForwardLayer_2_1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: t
2Encoder-FeedForwardLayer_2_1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
-Encoder-FeedForwardLayer_2_1/Tensordot/concatConcatV24Encoder-FeedForwardLayer_2_1/Tensordot/free:output:04Encoder-FeedForwardLayer_2_1/Tensordot/axes:output:0;Encoder-FeedForwardLayer_2_1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
,Encoder-FeedForwardLayer_2_1/Tensordot/stackPack4Encoder-FeedForwardLayer_2_1/Tensordot/Prod:output:06Encoder-FeedForwardLayer_2_1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
0Encoder-FeedForwardLayer_2_1/Tensordot/transpose	Transpose+Encoder-FeedForwardLayer_1_1/Gelu/mul_1:z:06Encoder-FeedForwardLayer_2_1/Tensordot/concat:output:0*
T0*+
_output_shapes
:���������P	�
.Encoder-FeedForwardLayer_2_1/Tensordot/ReshapeReshape4Encoder-FeedForwardLayer_2_1/Tensordot/transpose:y:05Encoder-FeedForwardLayer_2_1/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
-Encoder-FeedForwardLayer_2_1/Tensordot/MatMulMatMul7Encoder-FeedForwardLayer_2_1/Tensordot/Reshape:output:0=Encoder-FeedForwardLayer_2_1/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������	x
.Encoder-FeedForwardLayer_2_1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:	v
4Encoder-FeedForwardLayer_2_1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
/Encoder-FeedForwardLayer_2_1/Tensordot/concat_1ConcatV28Encoder-FeedForwardLayer_2_1/Tensordot/GatherV2:output:07Encoder-FeedForwardLayer_2_1/Tensordot/Const_2:output:0=Encoder-FeedForwardLayer_2_1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
&Encoder-FeedForwardLayer_2_1/TensordotReshape7Encoder-FeedForwardLayer_2_1/Tensordot/MatMul:product:08Encoder-FeedForwardLayer_2_1/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������P	�
3Encoder-FeedForwardLayer_2_1/BiasAdd/ReadVariableOpReadVariableOp<encoder_feedforwardlayer_2_1_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype0�
$Encoder-FeedForwardLayer_2_1/BiasAddBiasAdd/Encoder-FeedForwardLayer_2_1/Tensordot:output:0;Encoder-FeedForwardLayer_2_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������P	�
dropout_3/IdentityIdentity-Encoder-FeedForwardLayer_2_1/BiasAdd:output:0*
T0*+
_output_shapes
:���������P	�
Encoder-2nd-AdditionLayer-1/addAddV2#Encoder-1st-AdditionLayer-1/add:z:0dropout_3/Identity:output:0*
T0*+
_output_shapes
:���������P	v
IdentityIdentity#Encoder-2nd-AdditionLayer-1/add:z:0^NoOp*
T0*+
_output_shapes
:���������P	�

Identity_1Identity6Encoder-SelfAttentionLayer-1/softmax/Softmax:softmax:0^NoOp*
T0*/
_output_shapes
:���������PP�
NoOpNoOp4^Encoder-1st-NormalizationLayer-1/add/ReadVariableOp4^Encoder-1st-NormalizationLayer-1/mul/ReadVariableOp4^Encoder-2nd-NormalizationLayer-1/add/ReadVariableOp4^Encoder-2nd-NormalizationLayer-1/mul/ReadVariableOp4^Encoder-FeedForwardLayer_1_1/BiasAdd/ReadVariableOp6^Encoder-FeedForwardLayer_1_1/Tensordot/ReadVariableOp4^Encoder-FeedForwardLayer_2_1/BiasAdd/ReadVariableOp6^Encoder-FeedForwardLayer_2_1/Tensordot/ReadVariableOpA^Encoder-SelfAttentionLayer-1/attention_output/add/ReadVariableOpK^Encoder-SelfAttentionLayer-1/attention_output/einsum/Einsum/ReadVariableOp4^Encoder-SelfAttentionLayer-1/key/add/ReadVariableOp>^Encoder-SelfAttentionLayer-1/key/einsum/Einsum/ReadVariableOp6^Encoder-SelfAttentionLayer-1/query/add/ReadVariableOp@^Encoder-SelfAttentionLayer-1/query/einsum/Einsum/ReadVariableOp6^Encoder-SelfAttentionLayer-1/value/add/ReadVariableOp@^Encoder-SelfAttentionLayer-1/value/einsum/Einsum/ReadVariableOp*
_output_shapes
 "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:���������P	: : : : : : : : : : : : : : : : 2j
3Encoder-1st-NormalizationLayer-1/add/ReadVariableOp3Encoder-1st-NormalizationLayer-1/add/ReadVariableOp2j
3Encoder-1st-NormalizationLayer-1/mul/ReadVariableOp3Encoder-1st-NormalizationLayer-1/mul/ReadVariableOp2j
3Encoder-2nd-NormalizationLayer-1/add/ReadVariableOp3Encoder-2nd-NormalizationLayer-1/add/ReadVariableOp2j
3Encoder-2nd-NormalizationLayer-1/mul/ReadVariableOp3Encoder-2nd-NormalizationLayer-1/mul/ReadVariableOp2j
3Encoder-FeedForwardLayer_1_1/BiasAdd/ReadVariableOp3Encoder-FeedForwardLayer_1_1/BiasAdd/ReadVariableOp2n
5Encoder-FeedForwardLayer_1_1/Tensordot/ReadVariableOp5Encoder-FeedForwardLayer_1_1/Tensordot/ReadVariableOp2j
3Encoder-FeedForwardLayer_2_1/BiasAdd/ReadVariableOp3Encoder-FeedForwardLayer_2_1/BiasAdd/ReadVariableOp2n
5Encoder-FeedForwardLayer_2_1/Tensordot/ReadVariableOp5Encoder-FeedForwardLayer_2_1/Tensordot/ReadVariableOp2�
@Encoder-SelfAttentionLayer-1/attention_output/add/ReadVariableOp@Encoder-SelfAttentionLayer-1/attention_output/add/ReadVariableOp2�
JEncoder-SelfAttentionLayer-1/attention_output/einsum/Einsum/ReadVariableOpJEncoder-SelfAttentionLayer-1/attention_output/einsum/Einsum/ReadVariableOp2j
3Encoder-SelfAttentionLayer-1/key/add/ReadVariableOp3Encoder-SelfAttentionLayer-1/key/add/ReadVariableOp2~
=Encoder-SelfAttentionLayer-1/key/einsum/Einsum/ReadVariableOp=Encoder-SelfAttentionLayer-1/key/einsum/Einsum/ReadVariableOp2n
5Encoder-SelfAttentionLayer-1/query/add/ReadVariableOp5Encoder-SelfAttentionLayer-1/query/add/ReadVariableOp2�
?Encoder-SelfAttentionLayer-1/query/einsum/Einsum/ReadVariableOp?Encoder-SelfAttentionLayer-1/query/einsum/Einsum/ReadVariableOp2n
5Encoder-SelfAttentionLayer-1/value/add/ReadVariableOp5Encoder-SelfAttentionLayer-1/value/add/ReadVariableOp2�
?Encoder-SelfAttentionLayer-1/value/einsum/Einsum/ReadVariableOp?Encoder-SelfAttentionLayer-1/value/einsum/Einsum/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:(
$
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:S O
+
_output_shapes
:���������P	
 
_user_specified_nameinputs
�
Q
5__inference_StandardizeTimeLimit_layer_call_fn_233291

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *Y
fTRR
P__inference_StandardizeTimeLimit_layer_call_and_return_conditional_losses_230500`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
^
B__inference_Output_layer_call_and_return_conditional_losses_230553

inputs
identity=
SqueezeSqueezeinputs*
T0*
_output_shapes
:I
IdentityIdentitySqueeze:output:0*
T0*
_output_shapes
:"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
��
�
Q__inference_transformer_encoder_4_layer_call_and_return_conditional_losses_232769

inputsJ
<encoder_1st_normalizationlayer_2_mul_readvariableop_resource:	J
<encoder_1st_normalizationlayer_2_add_readvariableop_resource:	^
Hencoder_selfattentionlayer_2_query_einsum_einsum_readvariableop_resource:	P
>encoder_selfattentionlayer_2_query_add_readvariableop_resource:\
Fencoder_selfattentionlayer_2_key_einsum_einsum_readvariableop_resource:	N
<encoder_selfattentionlayer_2_key_add_readvariableop_resource:^
Hencoder_selfattentionlayer_2_value_einsum_einsum_readvariableop_resource:	P
>encoder_selfattentionlayer_2_value_add_readvariableop_resource:i
Sencoder_selfattentionlayer_2_attention_output_einsum_einsum_readvariableop_resource:	W
Iencoder_selfattentionlayer_2_attention_output_add_readvariableop_resource:	J
<encoder_2nd_normalizationlayer_2_mul_readvariableop_resource:	J
<encoder_2nd_normalizationlayer_2_add_readvariableop_resource:	P
>encoder_feedforwardlayer_1_2_tensordot_readvariableop_resource:		J
<encoder_feedforwardlayer_1_2_biasadd_readvariableop_resource:	P
>encoder_feedforwardlayer_2_2_tensordot_readvariableop_resource:		J
<encoder_feedforwardlayer_2_2_biasadd_readvariableop_resource:	
identity

identity_1��3Encoder-1st-NormalizationLayer-2/add/ReadVariableOp�3Encoder-1st-NormalizationLayer-2/mul/ReadVariableOp�3Encoder-2nd-NormalizationLayer-2/add/ReadVariableOp�3Encoder-2nd-NormalizationLayer-2/mul/ReadVariableOp�3Encoder-FeedForwardLayer_1_2/BiasAdd/ReadVariableOp�5Encoder-FeedForwardLayer_1_2/Tensordot/ReadVariableOp�3Encoder-FeedForwardLayer_2_2/BiasAdd/ReadVariableOp�5Encoder-FeedForwardLayer_2_2/Tensordot/ReadVariableOp�@Encoder-SelfAttentionLayer-2/attention_output/add/ReadVariableOp�JEncoder-SelfAttentionLayer-2/attention_output/einsum/Einsum/ReadVariableOp�3Encoder-SelfAttentionLayer-2/key/add/ReadVariableOp�=Encoder-SelfAttentionLayer-2/key/einsum/Einsum/ReadVariableOp�5Encoder-SelfAttentionLayer-2/query/add/ReadVariableOp�?Encoder-SelfAttentionLayer-2/query/einsum/Einsum/ReadVariableOp�5Encoder-SelfAttentionLayer-2/value/add/ReadVariableOp�?Encoder-SelfAttentionLayer-2/value/einsum/Einsum/ReadVariableOpj
&Encoder-1st-NormalizationLayer-2/ShapeShapeinputs*
T0*
_output_shapes
::��~
4Encoder-1st-NormalizationLayer-2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
6Encoder-1st-NormalizationLayer-2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
6Encoder-1st-NormalizationLayer-2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
.Encoder-1st-NormalizationLayer-2/strided_sliceStridedSlice/Encoder-1st-NormalizationLayer-2/Shape:output:0=Encoder-1st-NormalizationLayer-2/strided_slice/stack:output:0?Encoder-1st-NormalizationLayer-2/strided_slice/stack_1:output:0?Encoder-1st-NormalizationLayer-2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskp
&Encoder-1st-NormalizationLayer-2/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
%Encoder-1st-NormalizationLayer-2/ProdProd7Encoder-1st-NormalizationLayer-2/strided_slice:output:0/Encoder-1st-NormalizationLayer-2/Const:output:0*
T0*
_output_shapes
: �
6Encoder-1st-NormalizationLayer-2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:�
8Encoder-1st-NormalizationLayer-2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: �
8Encoder-1st-NormalizationLayer-2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
0Encoder-1st-NormalizationLayer-2/strided_slice_1StridedSlice/Encoder-1st-NormalizationLayer-2/Shape:output:0?Encoder-1st-NormalizationLayer-2/strided_slice_1/stack:output:0AEncoder-1st-NormalizationLayer-2/strided_slice_1/stack_1:output:0AEncoder-1st-NormalizationLayer-2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_maskr
(Encoder-1st-NormalizationLayer-2/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
'Encoder-1st-NormalizationLayer-2/Prod_1Prod9Encoder-1st-NormalizationLayer-2/strided_slice_1:output:01Encoder-1st-NormalizationLayer-2/Const_1:output:0*
T0*
_output_shapes
: r
0Encoder-1st-NormalizationLayer-2/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :r
0Encoder-1st-NormalizationLayer-2/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :�
.Encoder-1st-NormalizationLayer-2/Reshape/shapePack9Encoder-1st-NormalizationLayer-2/Reshape/shape/0:output:0.Encoder-1st-NormalizationLayer-2/Prod:output:00Encoder-1st-NormalizationLayer-2/Prod_1:output:09Encoder-1st-NormalizationLayer-2/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:�
(Encoder-1st-NormalizationLayer-2/ReshapeReshapeinputs7Encoder-1st-NormalizationLayer-2/Reshape/shape:output:0*
T0*8
_output_shapes&
$:"�������������������
,Encoder-1st-NormalizationLayer-2/ones/packedPack.Encoder-1st-NormalizationLayer-2/Prod:output:0*
N*
T0*
_output_shapes
:p
+Encoder-1st-NormalizationLayer-2/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
%Encoder-1st-NormalizationLayer-2/onesFill5Encoder-1st-NormalizationLayer-2/ones/packed:output:04Encoder-1st-NormalizationLayer-2/ones/Const:output:0*
T0*#
_output_shapes
:����������
-Encoder-1st-NormalizationLayer-2/zeros/packedPack.Encoder-1st-NormalizationLayer-2/Prod:output:0*
N*
T0*
_output_shapes
:q
,Encoder-1st-NormalizationLayer-2/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
&Encoder-1st-NormalizationLayer-2/zerosFill6Encoder-1st-NormalizationLayer-2/zeros/packed:output:05Encoder-1st-NormalizationLayer-2/zeros/Const:output:0*
T0*#
_output_shapes
:���������k
(Encoder-1st-NormalizationLayer-2/Const_2Const*
_output_shapes
: *
dtype0*
valueB k
(Encoder-1st-NormalizationLayer-2/Const_3Const*
_output_shapes
: *
dtype0*
valueB �
1Encoder-1st-NormalizationLayer-2/FusedBatchNormV3FusedBatchNormV31Encoder-1st-NormalizationLayer-2/Reshape:output:0.Encoder-1st-NormalizationLayer-2/ones:output:0/Encoder-1st-NormalizationLayer-2/zeros:output:01Encoder-1st-NormalizationLayer-2/Const_2:output:01Encoder-1st-NormalizationLayer-2/Const_3:output:0*
T0*
U0*x
_output_shapesf
d:"������������������:���������:���������:���������:���������:*
data_formatNCHW*
epsilon%o�:�
*Encoder-1st-NormalizationLayer-2/Reshape_1Reshape5Encoder-1st-NormalizationLayer-2/FusedBatchNormV3:y:0/Encoder-1st-NormalizationLayer-2/Shape:output:0*
T0*+
_output_shapes
:���������P	�
3Encoder-1st-NormalizationLayer-2/mul/ReadVariableOpReadVariableOp<encoder_1st_normalizationlayer_2_mul_readvariableop_resource*
_output_shapes
:	*
dtype0�
$Encoder-1st-NormalizationLayer-2/mulMul3Encoder-1st-NormalizationLayer-2/Reshape_1:output:0;Encoder-1st-NormalizationLayer-2/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������P	�
3Encoder-1st-NormalizationLayer-2/add/ReadVariableOpReadVariableOp<encoder_1st_normalizationlayer_2_add_readvariableop_resource*
_output_shapes
:	*
dtype0�
$Encoder-1st-NormalizationLayer-2/addAddV2(Encoder-1st-NormalizationLayer-2/mul:z:0;Encoder-1st-NormalizationLayer-2/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������P	�
?Encoder-SelfAttentionLayer-2/query/einsum/Einsum/ReadVariableOpReadVariableOpHencoder_selfattentionlayer_2_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:	*
dtype0�
0Encoder-SelfAttentionLayer-2/query/einsum/EinsumEinsum(Encoder-1st-NormalizationLayer-2/add:z:0GEncoder-SelfAttentionLayer-2/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������P*
equationabc,cde->abde�
5Encoder-SelfAttentionLayer-2/query/add/ReadVariableOpReadVariableOp>encoder_selfattentionlayer_2_query_add_readvariableop_resource*
_output_shapes

:*
dtype0�
&Encoder-SelfAttentionLayer-2/query/addAddV29Encoder-SelfAttentionLayer-2/query/einsum/Einsum:output:0=Encoder-SelfAttentionLayer-2/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������P�
=Encoder-SelfAttentionLayer-2/key/einsum/Einsum/ReadVariableOpReadVariableOpFencoder_selfattentionlayer_2_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:	*
dtype0�
.Encoder-SelfAttentionLayer-2/key/einsum/EinsumEinsum(Encoder-1st-NormalizationLayer-2/add:z:0EEncoder-SelfAttentionLayer-2/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������P*
equationabc,cde->abde�
3Encoder-SelfAttentionLayer-2/key/add/ReadVariableOpReadVariableOp<encoder_selfattentionlayer_2_key_add_readvariableop_resource*
_output_shapes

:*
dtype0�
$Encoder-SelfAttentionLayer-2/key/addAddV27Encoder-SelfAttentionLayer-2/key/einsum/Einsum:output:0;Encoder-SelfAttentionLayer-2/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������P�
?Encoder-SelfAttentionLayer-2/value/einsum/Einsum/ReadVariableOpReadVariableOpHencoder_selfattentionlayer_2_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:	*
dtype0�
0Encoder-SelfAttentionLayer-2/value/einsum/EinsumEinsum(Encoder-1st-NormalizationLayer-2/add:z:0GEncoder-SelfAttentionLayer-2/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������P*
equationabc,cde->abde�
5Encoder-SelfAttentionLayer-2/value/add/ReadVariableOpReadVariableOp>encoder_selfattentionlayer_2_value_add_readvariableop_resource*
_output_shapes

:*
dtype0�
&Encoder-SelfAttentionLayer-2/value/addAddV29Encoder-SelfAttentionLayer-2/value/einsum/Einsum:output:0=Encoder-SelfAttentionLayer-2/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������Pg
"Encoder-SelfAttentionLayer-2/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *:�?�
 Encoder-SelfAttentionLayer-2/MulMul*Encoder-SelfAttentionLayer-2/query/add:z:0+Encoder-SelfAttentionLayer-2/Mul/y:output:0*
T0*/
_output_shapes
:���������P�
*Encoder-SelfAttentionLayer-2/einsum/EinsumEinsum(Encoder-SelfAttentionLayer-2/key/add:z:0$Encoder-SelfAttentionLayer-2/Mul:z:0*
N*
T0*/
_output_shapes
:���������PP*
equationaecd,abcd->acbe�
,Encoder-SelfAttentionLayer-2/softmax/SoftmaxSoftmax3Encoder-SelfAttentionLayer-2/einsum/Einsum:output:0*
T0*/
_output_shapes
:���������PP�
-Encoder-SelfAttentionLayer-2/dropout/IdentityIdentity6Encoder-SelfAttentionLayer-2/softmax/Softmax:softmax:0*
T0*/
_output_shapes
:���������PP�
,Encoder-SelfAttentionLayer-2/einsum_1/EinsumEinsum6Encoder-SelfAttentionLayer-2/dropout/Identity:output:0*Encoder-SelfAttentionLayer-2/value/add:z:0*
N*
T0*/
_output_shapes
:���������P*
equationacbe,aecd->abcd�
JEncoder-SelfAttentionLayer-2/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpSencoder_selfattentionlayer_2_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:	*
dtype0�
;Encoder-SelfAttentionLayer-2/attention_output/einsum/EinsumEinsum5Encoder-SelfAttentionLayer-2/einsum_1/Einsum:output:0REncoder-SelfAttentionLayer-2/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:���������P	*
equationabcd,cde->abe�
@Encoder-SelfAttentionLayer-2/attention_output/add/ReadVariableOpReadVariableOpIencoder_selfattentionlayer_2_attention_output_add_readvariableop_resource*
_output_shapes
:	*
dtype0�
1Encoder-SelfAttentionLayer-2/attention_output/addAddV2DEncoder-SelfAttentionLayer-2/attention_output/einsum/Einsum:output:0HEncoder-SelfAttentionLayer-2/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������P	�
Encoder-1st-AdditionLayer-2/addAddV2inputs5Encoder-SelfAttentionLayer-2/attention_output/add:z:0*
T0*+
_output_shapes
:���������P	�
&Encoder-2nd-NormalizationLayer-2/ShapeShape#Encoder-1st-AdditionLayer-2/add:z:0*
T0*
_output_shapes
::��~
4Encoder-2nd-NormalizationLayer-2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
6Encoder-2nd-NormalizationLayer-2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
6Encoder-2nd-NormalizationLayer-2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
.Encoder-2nd-NormalizationLayer-2/strided_sliceStridedSlice/Encoder-2nd-NormalizationLayer-2/Shape:output:0=Encoder-2nd-NormalizationLayer-2/strided_slice/stack:output:0?Encoder-2nd-NormalizationLayer-2/strided_slice/stack_1:output:0?Encoder-2nd-NormalizationLayer-2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskp
&Encoder-2nd-NormalizationLayer-2/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
%Encoder-2nd-NormalizationLayer-2/ProdProd7Encoder-2nd-NormalizationLayer-2/strided_slice:output:0/Encoder-2nd-NormalizationLayer-2/Const:output:0*
T0*
_output_shapes
: �
6Encoder-2nd-NormalizationLayer-2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:�
8Encoder-2nd-NormalizationLayer-2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: �
8Encoder-2nd-NormalizationLayer-2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
0Encoder-2nd-NormalizationLayer-2/strided_slice_1StridedSlice/Encoder-2nd-NormalizationLayer-2/Shape:output:0?Encoder-2nd-NormalizationLayer-2/strided_slice_1/stack:output:0AEncoder-2nd-NormalizationLayer-2/strided_slice_1/stack_1:output:0AEncoder-2nd-NormalizationLayer-2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_maskr
(Encoder-2nd-NormalizationLayer-2/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
'Encoder-2nd-NormalizationLayer-2/Prod_1Prod9Encoder-2nd-NormalizationLayer-2/strided_slice_1:output:01Encoder-2nd-NormalizationLayer-2/Const_1:output:0*
T0*
_output_shapes
: r
0Encoder-2nd-NormalizationLayer-2/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :r
0Encoder-2nd-NormalizationLayer-2/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :�
.Encoder-2nd-NormalizationLayer-2/Reshape/shapePack9Encoder-2nd-NormalizationLayer-2/Reshape/shape/0:output:0.Encoder-2nd-NormalizationLayer-2/Prod:output:00Encoder-2nd-NormalizationLayer-2/Prod_1:output:09Encoder-2nd-NormalizationLayer-2/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:�
(Encoder-2nd-NormalizationLayer-2/ReshapeReshape#Encoder-1st-AdditionLayer-2/add:z:07Encoder-2nd-NormalizationLayer-2/Reshape/shape:output:0*
T0*8
_output_shapes&
$:"�������������������
,Encoder-2nd-NormalizationLayer-2/ones/packedPack.Encoder-2nd-NormalizationLayer-2/Prod:output:0*
N*
T0*
_output_shapes
:p
+Encoder-2nd-NormalizationLayer-2/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
%Encoder-2nd-NormalizationLayer-2/onesFill5Encoder-2nd-NormalizationLayer-2/ones/packed:output:04Encoder-2nd-NormalizationLayer-2/ones/Const:output:0*
T0*#
_output_shapes
:����������
-Encoder-2nd-NormalizationLayer-2/zeros/packedPack.Encoder-2nd-NormalizationLayer-2/Prod:output:0*
N*
T0*
_output_shapes
:q
,Encoder-2nd-NormalizationLayer-2/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
&Encoder-2nd-NormalizationLayer-2/zerosFill6Encoder-2nd-NormalizationLayer-2/zeros/packed:output:05Encoder-2nd-NormalizationLayer-2/zeros/Const:output:0*
T0*#
_output_shapes
:���������k
(Encoder-2nd-NormalizationLayer-2/Const_2Const*
_output_shapes
: *
dtype0*
valueB k
(Encoder-2nd-NormalizationLayer-2/Const_3Const*
_output_shapes
: *
dtype0*
valueB �
1Encoder-2nd-NormalizationLayer-2/FusedBatchNormV3FusedBatchNormV31Encoder-2nd-NormalizationLayer-2/Reshape:output:0.Encoder-2nd-NormalizationLayer-2/ones:output:0/Encoder-2nd-NormalizationLayer-2/zeros:output:01Encoder-2nd-NormalizationLayer-2/Const_2:output:01Encoder-2nd-NormalizationLayer-2/Const_3:output:0*
T0*
U0*x
_output_shapesf
d:"������������������:���������:���������:���������:���������:*
data_formatNCHW*
epsilon%o�:�
*Encoder-2nd-NormalizationLayer-2/Reshape_1Reshape5Encoder-2nd-NormalizationLayer-2/FusedBatchNormV3:y:0/Encoder-2nd-NormalizationLayer-2/Shape:output:0*
T0*+
_output_shapes
:���������P	�
3Encoder-2nd-NormalizationLayer-2/mul/ReadVariableOpReadVariableOp<encoder_2nd_normalizationlayer_2_mul_readvariableop_resource*
_output_shapes
:	*
dtype0�
$Encoder-2nd-NormalizationLayer-2/mulMul3Encoder-2nd-NormalizationLayer-2/Reshape_1:output:0;Encoder-2nd-NormalizationLayer-2/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������P	�
3Encoder-2nd-NormalizationLayer-2/add/ReadVariableOpReadVariableOp<encoder_2nd_normalizationlayer_2_add_readvariableop_resource*
_output_shapes
:	*
dtype0�
$Encoder-2nd-NormalizationLayer-2/addAddV2(Encoder-2nd-NormalizationLayer-2/mul:z:0;Encoder-2nd-NormalizationLayer-2/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������P	�
5Encoder-FeedForwardLayer_1_2/Tensordot/ReadVariableOpReadVariableOp>encoder_feedforwardlayer_1_2_tensordot_readvariableop_resource*
_output_shapes

:		*
dtype0u
+Encoder-FeedForwardLayer_1_2/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:|
+Encoder-FeedForwardLayer_1_2/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       �
,Encoder-FeedForwardLayer_1_2/Tensordot/ShapeShape(Encoder-2nd-NormalizationLayer-2/add:z:0*
T0*
_output_shapes
::��v
4Encoder-FeedForwardLayer_1_2/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
/Encoder-FeedForwardLayer_1_2/Tensordot/GatherV2GatherV25Encoder-FeedForwardLayer_1_2/Tensordot/Shape:output:04Encoder-FeedForwardLayer_1_2/Tensordot/free:output:0=Encoder-FeedForwardLayer_1_2/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:x
6Encoder-FeedForwardLayer_1_2/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
1Encoder-FeedForwardLayer_1_2/Tensordot/GatherV2_1GatherV25Encoder-FeedForwardLayer_1_2/Tensordot/Shape:output:04Encoder-FeedForwardLayer_1_2/Tensordot/axes:output:0?Encoder-FeedForwardLayer_1_2/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:v
,Encoder-FeedForwardLayer_1_2/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
+Encoder-FeedForwardLayer_1_2/Tensordot/ProdProd8Encoder-FeedForwardLayer_1_2/Tensordot/GatherV2:output:05Encoder-FeedForwardLayer_1_2/Tensordot/Const:output:0*
T0*
_output_shapes
: x
.Encoder-FeedForwardLayer_1_2/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
-Encoder-FeedForwardLayer_1_2/Tensordot/Prod_1Prod:Encoder-FeedForwardLayer_1_2/Tensordot/GatherV2_1:output:07Encoder-FeedForwardLayer_1_2/Tensordot/Const_1:output:0*
T0*
_output_shapes
: t
2Encoder-FeedForwardLayer_1_2/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
-Encoder-FeedForwardLayer_1_2/Tensordot/concatConcatV24Encoder-FeedForwardLayer_1_2/Tensordot/free:output:04Encoder-FeedForwardLayer_1_2/Tensordot/axes:output:0;Encoder-FeedForwardLayer_1_2/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
,Encoder-FeedForwardLayer_1_2/Tensordot/stackPack4Encoder-FeedForwardLayer_1_2/Tensordot/Prod:output:06Encoder-FeedForwardLayer_1_2/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
0Encoder-FeedForwardLayer_1_2/Tensordot/transpose	Transpose(Encoder-2nd-NormalizationLayer-2/add:z:06Encoder-FeedForwardLayer_1_2/Tensordot/concat:output:0*
T0*+
_output_shapes
:���������P	�
.Encoder-FeedForwardLayer_1_2/Tensordot/ReshapeReshape4Encoder-FeedForwardLayer_1_2/Tensordot/transpose:y:05Encoder-FeedForwardLayer_1_2/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
-Encoder-FeedForwardLayer_1_2/Tensordot/MatMulMatMul7Encoder-FeedForwardLayer_1_2/Tensordot/Reshape:output:0=Encoder-FeedForwardLayer_1_2/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������	x
.Encoder-FeedForwardLayer_1_2/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:	v
4Encoder-FeedForwardLayer_1_2/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
/Encoder-FeedForwardLayer_1_2/Tensordot/concat_1ConcatV28Encoder-FeedForwardLayer_1_2/Tensordot/GatherV2:output:07Encoder-FeedForwardLayer_1_2/Tensordot/Const_2:output:0=Encoder-FeedForwardLayer_1_2/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
&Encoder-FeedForwardLayer_1_2/TensordotReshape7Encoder-FeedForwardLayer_1_2/Tensordot/MatMul:product:08Encoder-FeedForwardLayer_1_2/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������P	�
3Encoder-FeedForwardLayer_1_2/BiasAdd/ReadVariableOpReadVariableOp<encoder_feedforwardlayer_1_2_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype0�
$Encoder-FeedForwardLayer_1_2/BiasAddBiasAdd/Encoder-FeedForwardLayer_1_2/Tensordot:output:0;Encoder-FeedForwardLayer_1_2/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������P	l
'Encoder-FeedForwardLayer_1_2/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
%Encoder-FeedForwardLayer_1_2/Gelu/mulMul0Encoder-FeedForwardLayer_1_2/Gelu/mul/x:output:0-Encoder-FeedForwardLayer_1_2/BiasAdd:output:0*
T0*+
_output_shapes
:���������P	m
(Encoder-FeedForwardLayer_1_2/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *��?�
)Encoder-FeedForwardLayer_1_2/Gelu/truedivRealDiv-Encoder-FeedForwardLayer_1_2/BiasAdd:output:01Encoder-FeedForwardLayer_1_2/Gelu/Cast/x:output:0*
T0*+
_output_shapes
:���������P	�
%Encoder-FeedForwardLayer_1_2/Gelu/ErfErf-Encoder-FeedForwardLayer_1_2/Gelu/truediv:z:0*
T0*+
_output_shapes
:���������P	l
'Encoder-FeedForwardLayer_1_2/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
%Encoder-FeedForwardLayer_1_2/Gelu/addAddV20Encoder-FeedForwardLayer_1_2/Gelu/add/x:output:0)Encoder-FeedForwardLayer_1_2/Gelu/Erf:y:0*
T0*+
_output_shapes
:���������P	�
'Encoder-FeedForwardLayer_1_2/Gelu/mul_1Mul)Encoder-FeedForwardLayer_1_2/Gelu/mul:z:0)Encoder-FeedForwardLayer_1_2/Gelu/add:z:0*
T0*+
_output_shapes
:���������P	�
5Encoder-FeedForwardLayer_2_2/Tensordot/ReadVariableOpReadVariableOp>encoder_feedforwardlayer_2_2_tensordot_readvariableop_resource*
_output_shapes

:		*
dtype0u
+Encoder-FeedForwardLayer_2_2/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:|
+Encoder-FeedForwardLayer_2_2/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       �
,Encoder-FeedForwardLayer_2_2/Tensordot/ShapeShape+Encoder-FeedForwardLayer_1_2/Gelu/mul_1:z:0*
T0*
_output_shapes
::��v
4Encoder-FeedForwardLayer_2_2/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
/Encoder-FeedForwardLayer_2_2/Tensordot/GatherV2GatherV25Encoder-FeedForwardLayer_2_2/Tensordot/Shape:output:04Encoder-FeedForwardLayer_2_2/Tensordot/free:output:0=Encoder-FeedForwardLayer_2_2/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:x
6Encoder-FeedForwardLayer_2_2/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
1Encoder-FeedForwardLayer_2_2/Tensordot/GatherV2_1GatherV25Encoder-FeedForwardLayer_2_2/Tensordot/Shape:output:04Encoder-FeedForwardLayer_2_2/Tensordot/axes:output:0?Encoder-FeedForwardLayer_2_2/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:v
,Encoder-FeedForwardLayer_2_2/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
+Encoder-FeedForwardLayer_2_2/Tensordot/ProdProd8Encoder-FeedForwardLayer_2_2/Tensordot/GatherV2:output:05Encoder-FeedForwardLayer_2_2/Tensordot/Const:output:0*
T0*
_output_shapes
: x
.Encoder-FeedForwardLayer_2_2/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
-Encoder-FeedForwardLayer_2_2/Tensordot/Prod_1Prod:Encoder-FeedForwardLayer_2_2/Tensordot/GatherV2_1:output:07Encoder-FeedForwardLayer_2_2/Tensordot/Const_1:output:0*
T0*
_output_shapes
: t
2Encoder-FeedForwardLayer_2_2/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
-Encoder-FeedForwardLayer_2_2/Tensordot/concatConcatV24Encoder-FeedForwardLayer_2_2/Tensordot/free:output:04Encoder-FeedForwardLayer_2_2/Tensordot/axes:output:0;Encoder-FeedForwardLayer_2_2/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
,Encoder-FeedForwardLayer_2_2/Tensordot/stackPack4Encoder-FeedForwardLayer_2_2/Tensordot/Prod:output:06Encoder-FeedForwardLayer_2_2/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
0Encoder-FeedForwardLayer_2_2/Tensordot/transpose	Transpose+Encoder-FeedForwardLayer_1_2/Gelu/mul_1:z:06Encoder-FeedForwardLayer_2_2/Tensordot/concat:output:0*
T0*+
_output_shapes
:���������P	�
.Encoder-FeedForwardLayer_2_2/Tensordot/ReshapeReshape4Encoder-FeedForwardLayer_2_2/Tensordot/transpose:y:05Encoder-FeedForwardLayer_2_2/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
-Encoder-FeedForwardLayer_2_2/Tensordot/MatMulMatMul7Encoder-FeedForwardLayer_2_2/Tensordot/Reshape:output:0=Encoder-FeedForwardLayer_2_2/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������	x
.Encoder-FeedForwardLayer_2_2/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:	v
4Encoder-FeedForwardLayer_2_2/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
/Encoder-FeedForwardLayer_2_2/Tensordot/concat_1ConcatV28Encoder-FeedForwardLayer_2_2/Tensordot/GatherV2:output:07Encoder-FeedForwardLayer_2_2/Tensordot/Const_2:output:0=Encoder-FeedForwardLayer_2_2/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
&Encoder-FeedForwardLayer_2_2/TensordotReshape7Encoder-FeedForwardLayer_2_2/Tensordot/MatMul:product:08Encoder-FeedForwardLayer_2_2/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������P	�
3Encoder-FeedForwardLayer_2_2/BiasAdd/ReadVariableOpReadVariableOp<encoder_feedforwardlayer_2_2_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype0�
$Encoder-FeedForwardLayer_2_2/BiasAddBiasAdd/Encoder-FeedForwardLayer_2_2/Tensordot:output:0;Encoder-FeedForwardLayer_2_2/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������P	�
dropout_4/IdentityIdentity-Encoder-FeedForwardLayer_2_2/BiasAdd:output:0*
T0*+
_output_shapes
:���������P	�
Encoder-2nd-AdditionLayer-2/addAddV2#Encoder-1st-AdditionLayer-2/add:z:0dropout_4/Identity:output:0*
T0*+
_output_shapes
:���������P	v
IdentityIdentity#Encoder-2nd-AdditionLayer-2/add:z:0^NoOp*
T0*+
_output_shapes
:���������P	�

Identity_1Identity6Encoder-SelfAttentionLayer-2/softmax/Softmax:softmax:0^NoOp*
T0*/
_output_shapes
:���������PP�
NoOpNoOp4^Encoder-1st-NormalizationLayer-2/add/ReadVariableOp4^Encoder-1st-NormalizationLayer-2/mul/ReadVariableOp4^Encoder-2nd-NormalizationLayer-2/add/ReadVariableOp4^Encoder-2nd-NormalizationLayer-2/mul/ReadVariableOp4^Encoder-FeedForwardLayer_1_2/BiasAdd/ReadVariableOp6^Encoder-FeedForwardLayer_1_2/Tensordot/ReadVariableOp4^Encoder-FeedForwardLayer_2_2/BiasAdd/ReadVariableOp6^Encoder-FeedForwardLayer_2_2/Tensordot/ReadVariableOpA^Encoder-SelfAttentionLayer-2/attention_output/add/ReadVariableOpK^Encoder-SelfAttentionLayer-2/attention_output/einsum/Einsum/ReadVariableOp4^Encoder-SelfAttentionLayer-2/key/add/ReadVariableOp>^Encoder-SelfAttentionLayer-2/key/einsum/Einsum/ReadVariableOp6^Encoder-SelfAttentionLayer-2/query/add/ReadVariableOp@^Encoder-SelfAttentionLayer-2/query/einsum/Einsum/ReadVariableOp6^Encoder-SelfAttentionLayer-2/value/add/ReadVariableOp@^Encoder-SelfAttentionLayer-2/value/einsum/Einsum/ReadVariableOp*
_output_shapes
 "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:���������P	: : : : : : : : : : : : : : : : 2j
3Encoder-1st-NormalizationLayer-2/add/ReadVariableOp3Encoder-1st-NormalizationLayer-2/add/ReadVariableOp2j
3Encoder-1st-NormalizationLayer-2/mul/ReadVariableOp3Encoder-1st-NormalizationLayer-2/mul/ReadVariableOp2j
3Encoder-2nd-NormalizationLayer-2/add/ReadVariableOp3Encoder-2nd-NormalizationLayer-2/add/ReadVariableOp2j
3Encoder-2nd-NormalizationLayer-2/mul/ReadVariableOp3Encoder-2nd-NormalizationLayer-2/mul/ReadVariableOp2j
3Encoder-FeedForwardLayer_1_2/BiasAdd/ReadVariableOp3Encoder-FeedForwardLayer_1_2/BiasAdd/ReadVariableOp2n
5Encoder-FeedForwardLayer_1_2/Tensordot/ReadVariableOp5Encoder-FeedForwardLayer_1_2/Tensordot/ReadVariableOp2j
3Encoder-FeedForwardLayer_2_2/BiasAdd/ReadVariableOp3Encoder-FeedForwardLayer_2_2/BiasAdd/ReadVariableOp2n
5Encoder-FeedForwardLayer_2_2/Tensordot/ReadVariableOp5Encoder-FeedForwardLayer_2_2/Tensordot/ReadVariableOp2�
@Encoder-SelfAttentionLayer-2/attention_output/add/ReadVariableOp@Encoder-SelfAttentionLayer-2/attention_output/add/ReadVariableOp2�
JEncoder-SelfAttentionLayer-2/attention_output/einsum/Einsum/ReadVariableOpJEncoder-SelfAttentionLayer-2/attention_output/einsum/Einsum/ReadVariableOp2j
3Encoder-SelfAttentionLayer-2/key/add/ReadVariableOp3Encoder-SelfAttentionLayer-2/key/add/ReadVariableOp2~
=Encoder-SelfAttentionLayer-2/key/einsum/Einsum/ReadVariableOp=Encoder-SelfAttentionLayer-2/key/einsum/Einsum/ReadVariableOp2n
5Encoder-SelfAttentionLayer-2/query/add/ReadVariableOp5Encoder-SelfAttentionLayer-2/query/add/ReadVariableOp2�
?Encoder-SelfAttentionLayer-2/query/einsum/Einsum/ReadVariableOp?Encoder-SelfAttentionLayer-2/query/einsum/Einsum/ReadVariableOp2n
5Encoder-SelfAttentionLayer-2/value/add/ReadVariableOp5Encoder-SelfAttentionLayer-2/value/add/ReadVariableOp2�
?Encoder-SelfAttentionLayer-2/value/einsum/Einsum/ReadVariableOp?Encoder-SelfAttentionLayer-2/value/einsum/Einsum/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:(
$
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:S O
+
_output_shapes
:���������P	
 
_user_specified_nameinputs
��
�
Q__inference_transformer_encoder_3_layer_call_and_return_conditional_losses_229957

inputsJ
<encoder_1st_normalizationlayer_1_mul_readvariableop_resource:	J
<encoder_1st_normalizationlayer_1_add_readvariableop_resource:	^
Hencoder_selfattentionlayer_1_query_einsum_einsum_readvariableop_resource:	P
>encoder_selfattentionlayer_1_query_add_readvariableop_resource:\
Fencoder_selfattentionlayer_1_key_einsum_einsum_readvariableop_resource:	N
<encoder_selfattentionlayer_1_key_add_readvariableop_resource:^
Hencoder_selfattentionlayer_1_value_einsum_einsum_readvariableop_resource:	P
>encoder_selfattentionlayer_1_value_add_readvariableop_resource:i
Sencoder_selfattentionlayer_1_attention_output_einsum_einsum_readvariableop_resource:	W
Iencoder_selfattentionlayer_1_attention_output_add_readvariableop_resource:	J
<encoder_2nd_normalizationlayer_1_mul_readvariableop_resource:	J
<encoder_2nd_normalizationlayer_1_add_readvariableop_resource:	P
>encoder_feedforwardlayer_1_1_tensordot_readvariableop_resource:		J
<encoder_feedforwardlayer_1_1_biasadd_readvariableop_resource:	P
>encoder_feedforwardlayer_2_1_tensordot_readvariableop_resource:		J
<encoder_feedforwardlayer_2_1_biasadd_readvariableop_resource:	
identity

identity_1��3Encoder-1st-NormalizationLayer-1/add/ReadVariableOp�3Encoder-1st-NormalizationLayer-1/mul/ReadVariableOp�3Encoder-2nd-NormalizationLayer-1/add/ReadVariableOp�3Encoder-2nd-NormalizationLayer-1/mul/ReadVariableOp�3Encoder-FeedForwardLayer_1_1/BiasAdd/ReadVariableOp�5Encoder-FeedForwardLayer_1_1/Tensordot/ReadVariableOp�3Encoder-FeedForwardLayer_2_1/BiasAdd/ReadVariableOp�5Encoder-FeedForwardLayer_2_1/Tensordot/ReadVariableOp�@Encoder-SelfAttentionLayer-1/attention_output/add/ReadVariableOp�JEncoder-SelfAttentionLayer-1/attention_output/einsum/Einsum/ReadVariableOp�3Encoder-SelfAttentionLayer-1/key/add/ReadVariableOp�=Encoder-SelfAttentionLayer-1/key/einsum/Einsum/ReadVariableOp�5Encoder-SelfAttentionLayer-1/query/add/ReadVariableOp�?Encoder-SelfAttentionLayer-1/query/einsum/Einsum/ReadVariableOp�5Encoder-SelfAttentionLayer-1/value/add/ReadVariableOp�?Encoder-SelfAttentionLayer-1/value/einsum/Einsum/ReadVariableOpj
&Encoder-1st-NormalizationLayer-1/ShapeShapeinputs*
T0*
_output_shapes
::��~
4Encoder-1st-NormalizationLayer-1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
6Encoder-1st-NormalizationLayer-1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
6Encoder-1st-NormalizationLayer-1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
.Encoder-1st-NormalizationLayer-1/strided_sliceStridedSlice/Encoder-1st-NormalizationLayer-1/Shape:output:0=Encoder-1st-NormalizationLayer-1/strided_slice/stack:output:0?Encoder-1st-NormalizationLayer-1/strided_slice/stack_1:output:0?Encoder-1st-NormalizationLayer-1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskp
&Encoder-1st-NormalizationLayer-1/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
%Encoder-1st-NormalizationLayer-1/ProdProd7Encoder-1st-NormalizationLayer-1/strided_slice:output:0/Encoder-1st-NormalizationLayer-1/Const:output:0*
T0*
_output_shapes
: �
6Encoder-1st-NormalizationLayer-1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:�
8Encoder-1st-NormalizationLayer-1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: �
8Encoder-1st-NormalizationLayer-1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
0Encoder-1st-NormalizationLayer-1/strided_slice_1StridedSlice/Encoder-1st-NormalizationLayer-1/Shape:output:0?Encoder-1st-NormalizationLayer-1/strided_slice_1/stack:output:0AEncoder-1st-NormalizationLayer-1/strided_slice_1/stack_1:output:0AEncoder-1st-NormalizationLayer-1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_maskr
(Encoder-1st-NormalizationLayer-1/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
'Encoder-1st-NormalizationLayer-1/Prod_1Prod9Encoder-1st-NormalizationLayer-1/strided_slice_1:output:01Encoder-1st-NormalizationLayer-1/Const_1:output:0*
T0*
_output_shapes
: r
0Encoder-1st-NormalizationLayer-1/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :r
0Encoder-1st-NormalizationLayer-1/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :�
.Encoder-1st-NormalizationLayer-1/Reshape/shapePack9Encoder-1st-NormalizationLayer-1/Reshape/shape/0:output:0.Encoder-1st-NormalizationLayer-1/Prod:output:00Encoder-1st-NormalizationLayer-1/Prod_1:output:09Encoder-1st-NormalizationLayer-1/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:�
(Encoder-1st-NormalizationLayer-1/ReshapeReshapeinputs7Encoder-1st-NormalizationLayer-1/Reshape/shape:output:0*
T0*8
_output_shapes&
$:"�������������������
,Encoder-1st-NormalizationLayer-1/ones/packedPack.Encoder-1st-NormalizationLayer-1/Prod:output:0*
N*
T0*
_output_shapes
:p
+Encoder-1st-NormalizationLayer-1/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
%Encoder-1st-NormalizationLayer-1/onesFill5Encoder-1st-NormalizationLayer-1/ones/packed:output:04Encoder-1st-NormalizationLayer-1/ones/Const:output:0*
T0*#
_output_shapes
:����������
-Encoder-1st-NormalizationLayer-1/zeros/packedPack.Encoder-1st-NormalizationLayer-1/Prod:output:0*
N*
T0*
_output_shapes
:q
,Encoder-1st-NormalizationLayer-1/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
&Encoder-1st-NormalizationLayer-1/zerosFill6Encoder-1st-NormalizationLayer-1/zeros/packed:output:05Encoder-1st-NormalizationLayer-1/zeros/Const:output:0*
T0*#
_output_shapes
:���������k
(Encoder-1st-NormalizationLayer-1/Const_2Const*
_output_shapes
: *
dtype0*
valueB k
(Encoder-1st-NormalizationLayer-1/Const_3Const*
_output_shapes
: *
dtype0*
valueB �
1Encoder-1st-NormalizationLayer-1/FusedBatchNormV3FusedBatchNormV31Encoder-1st-NormalizationLayer-1/Reshape:output:0.Encoder-1st-NormalizationLayer-1/ones:output:0/Encoder-1st-NormalizationLayer-1/zeros:output:01Encoder-1st-NormalizationLayer-1/Const_2:output:01Encoder-1st-NormalizationLayer-1/Const_3:output:0*
T0*
U0*x
_output_shapesf
d:"������������������:���������:���������:���������:���������:*
data_formatNCHW*
epsilon%o�:�
*Encoder-1st-NormalizationLayer-1/Reshape_1Reshape5Encoder-1st-NormalizationLayer-1/FusedBatchNormV3:y:0/Encoder-1st-NormalizationLayer-1/Shape:output:0*
T0*+
_output_shapes
:���������P	�
3Encoder-1st-NormalizationLayer-1/mul/ReadVariableOpReadVariableOp<encoder_1st_normalizationlayer_1_mul_readvariableop_resource*
_output_shapes
:	*
dtype0�
$Encoder-1st-NormalizationLayer-1/mulMul3Encoder-1st-NormalizationLayer-1/Reshape_1:output:0;Encoder-1st-NormalizationLayer-1/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������P	�
3Encoder-1st-NormalizationLayer-1/add/ReadVariableOpReadVariableOp<encoder_1st_normalizationlayer_1_add_readvariableop_resource*
_output_shapes
:	*
dtype0�
$Encoder-1st-NormalizationLayer-1/addAddV2(Encoder-1st-NormalizationLayer-1/mul:z:0;Encoder-1st-NormalizationLayer-1/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������P	�
?Encoder-SelfAttentionLayer-1/query/einsum/Einsum/ReadVariableOpReadVariableOpHencoder_selfattentionlayer_1_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:	*
dtype0�
0Encoder-SelfAttentionLayer-1/query/einsum/EinsumEinsum(Encoder-1st-NormalizationLayer-1/add:z:0GEncoder-SelfAttentionLayer-1/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������P*
equationabc,cde->abde�
5Encoder-SelfAttentionLayer-1/query/add/ReadVariableOpReadVariableOp>encoder_selfattentionlayer_1_query_add_readvariableop_resource*
_output_shapes

:*
dtype0�
&Encoder-SelfAttentionLayer-1/query/addAddV29Encoder-SelfAttentionLayer-1/query/einsum/Einsum:output:0=Encoder-SelfAttentionLayer-1/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������P�
=Encoder-SelfAttentionLayer-1/key/einsum/Einsum/ReadVariableOpReadVariableOpFencoder_selfattentionlayer_1_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:	*
dtype0�
.Encoder-SelfAttentionLayer-1/key/einsum/EinsumEinsum(Encoder-1st-NormalizationLayer-1/add:z:0EEncoder-SelfAttentionLayer-1/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������P*
equationabc,cde->abde�
3Encoder-SelfAttentionLayer-1/key/add/ReadVariableOpReadVariableOp<encoder_selfattentionlayer_1_key_add_readvariableop_resource*
_output_shapes

:*
dtype0�
$Encoder-SelfAttentionLayer-1/key/addAddV27Encoder-SelfAttentionLayer-1/key/einsum/Einsum:output:0;Encoder-SelfAttentionLayer-1/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������P�
?Encoder-SelfAttentionLayer-1/value/einsum/Einsum/ReadVariableOpReadVariableOpHencoder_selfattentionlayer_1_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:	*
dtype0�
0Encoder-SelfAttentionLayer-1/value/einsum/EinsumEinsum(Encoder-1st-NormalizationLayer-1/add:z:0GEncoder-SelfAttentionLayer-1/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������P*
equationabc,cde->abde�
5Encoder-SelfAttentionLayer-1/value/add/ReadVariableOpReadVariableOp>encoder_selfattentionlayer_1_value_add_readvariableop_resource*
_output_shapes

:*
dtype0�
&Encoder-SelfAttentionLayer-1/value/addAddV29Encoder-SelfAttentionLayer-1/value/einsum/Einsum:output:0=Encoder-SelfAttentionLayer-1/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������Pg
"Encoder-SelfAttentionLayer-1/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *:�?�
 Encoder-SelfAttentionLayer-1/MulMul*Encoder-SelfAttentionLayer-1/query/add:z:0+Encoder-SelfAttentionLayer-1/Mul/y:output:0*
T0*/
_output_shapes
:���������P�
*Encoder-SelfAttentionLayer-1/einsum/EinsumEinsum(Encoder-SelfAttentionLayer-1/key/add:z:0$Encoder-SelfAttentionLayer-1/Mul:z:0*
N*
T0*/
_output_shapes
:���������PP*
equationaecd,abcd->acbe�
,Encoder-SelfAttentionLayer-1/softmax/SoftmaxSoftmax3Encoder-SelfAttentionLayer-1/einsum/Einsum:output:0*
T0*/
_output_shapes
:���������PPw
2Encoder-SelfAttentionLayer-1/dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?�
0Encoder-SelfAttentionLayer-1/dropout/dropout/MulMul6Encoder-SelfAttentionLayer-1/softmax/Softmax:softmax:0;Encoder-SelfAttentionLayer-1/dropout/dropout/Const:output:0*
T0*/
_output_shapes
:���������PP�
2Encoder-SelfAttentionLayer-1/dropout/dropout/ShapeShape6Encoder-SelfAttentionLayer-1/softmax/Softmax:softmax:0*
T0*
_output_shapes
::���
IEncoder-SelfAttentionLayer-1/dropout/dropout/random_uniform/RandomUniformRandomUniform;Encoder-SelfAttentionLayer-1/dropout/dropout/Shape:output:0*
T0*/
_output_shapes
:���������PP*
dtype0*
seed���
;Encoder-SelfAttentionLayer-1/dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=�
9Encoder-SelfAttentionLayer-1/dropout/dropout/GreaterEqualGreaterEqualREncoder-SelfAttentionLayer-1/dropout/dropout/random_uniform/RandomUniform:output:0DEncoder-SelfAttentionLayer-1/dropout/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:���������PPy
4Encoder-SelfAttentionLayer-1/dropout/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
5Encoder-SelfAttentionLayer-1/dropout/dropout/SelectV2SelectV2=Encoder-SelfAttentionLayer-1/dropout/dropout/GreaterEqual:z:04Encoder-SelfAttentionLayer-1/dropout/dropout/Mul:z:0=Encoder-SelfAttentionLayer-1/dropout/dropout/Const_1:output:0*
T0*/
_output_shapes
:���������PP�
,Encoder-SelfAttentionLayer-1/einsum_1/EinsumEinsum>Encoder-SelfAttentionLayer-1/dropout/dropout/SelectV2:output:0*Encoder-SelfAttentionLayer-1/value/add:z:0*
N*
T0*/
_output_shapes
:���������P*
equationacbe,aecd->abcd�
JEncoder-SelfAttentionLayer-1/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpSencoder_selfattentionlayer_1_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:	*
dtype0�
;Encoder-SelfAttentionLayer-1/attention_output/einsum/EinsumEinsum5Encoder-SelfAttentionLayer-1/einsum_1/Einsum:output:0REncoder-SelfAttentionLayer-1/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:���������P	*
equationabcd,cde->abe�
@Encoder-SelfAttentionLayer-1/attention_output/add/ReadVariableOpReadVariableOpIencoder_selfattentionlayer_1_attention_output_add_readvariableop_resource*
_output_shapes
:	*
dtype0�
1Encoder-SelfAttentionLayer-1/attention_output/addAddV2DEncoder-SelfAttentionLayer-1/attention_output/einsum/Einsum:output:0HEncoder-SelfAttentionLayer-1/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������P	�
Encoder-1st-AdditionLayer-1/addAddV2inputs5Encoder-SelfAttentionLayer-1/attention_output/add:z:0*
T0*+
_output_shapes
:���������P	�
&Encoder-2nd-NormalizationLayer-1/ShapeShape#Encoder-1st-AdditionLayer-1/add:z:0*
T0*
_output_shapes
::��~
4Encoder-2nd-NormalizationLayer-1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
6Encoder-2nd-NormalizationLayer-1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
6Encoder-2nd-NormalizationLayer-1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
.Encoder-2nd-NormalizationLayer-1/strided_sliceStridedSlice/Encoder-2nd-NormalizationLayer-1/Shape:output:0=Encoder-2nd-NormalizationLayer-1/strided_slice/stack:output:0?Encoder-2nd-NormalizationLayer-1/strided_slice/stack_1:output:0?Encoder-2nd-NormalizationLayer-1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskp
&Encoder-2nd-NormalizationLayer-1/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
%Encoder-2nd-NormalizationLayer-1/ProdProd7Encoder-2nd-NormalizationLayer-1/strided_slice:output:0/Encoder-2nd-NormalizationLayer-1/Const:output:0*
T0*
_output_shapes
: �
6Encoder-2nd-NormalizationLayer-1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:�
8Encoder-2nd-NormalizationLayer-1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: �
8Encoder-2nd-NormalizationLayer-1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
0Encoder-2nd-NormalizationLayer-1/strided_slice_1StridedSlice/Encoder-2nd-NormalizationLayer-1/Shape:output:0?Encoder-2nd-NormalizationLayer-1/strided_slice_1/stack:output:0AEncoder-2nd-NormalizationLayer-1/strided_slice_1/stack_1:output:0AEncoder-2nd-NormalizationLayer-1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_maskr
(Encoder-2nd-NormalizationLayer-1/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
'Encoder-2nd-NormalizationLayer-1/Prod_1Prod9Encoder-2nd-NormalizationLayer-1/strided_slice_1:output:01Encoder-2nd-NormalizationLayer-1/Const_1:output:0*
T0*
_output_shapes
: r
0Encoder-2nd-NormalizationLayer-1/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :r
0Encoder-2nd-NormalizationLayer-1/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :�
.Encoder-2nd-NormalizationLayer-1/Reshape/shapePack9Encoder-2nd-NormalizationLayer-1/Reshape/shape/0:output:0.Encoder-2nd-NormalizationLayer-1/Prod:output:00Encoder-2nd-NormalizationLayer-1/Prod_1:output:09Encoder-2nd-NormalizationLayer-1/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:�
(Encoder-2nd-NormalizationLayer-1/ReshapeReshape#Encoder-1st-AdditionLayer-1/add:z:07Encoder-2nd-NormalizationLayer-1/Reshape/shape:output:0*
T0*8
_output_shapes&
$:"�������������������
,Encoder-2nd-NormalizationLayer-1/ones/packedPack.Encoder-2nd-NormalizationLayer-1/Prod:output:0*
N*
T0*
_output_shapes
:p
+Encoder-2nd-NormalizationLayer-1/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
%Encoder-2nd-NormalizationLayer-1/onesFill5Encoder-2nd-NormalizationLayer-1/ones/packed:output:04Encoder-2nd-NormalizationLayer-1/ones/Const:output:0*
T0*#
_output_shapes
:����������
-Encoder-2nd-NormalizationLayer-1/zeros/packedPack.Encoder-2nd-NormalizationLayer-1/Prod:output:0*
N*
T0*
_output_shapes
:q
,Encoder-2nd-NormalizationLayer-1/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
&Encoder-2nd-NormalizationLayer-1/zerosFill6Encoder-2nd-NormalizationLayer-1/zeros/packed:output:05Encoder-2nd-NormalizationLayer-1/zeros/Const:output:0*
T0*#
_output_shapes
:���������k
(Encoder-2nd-NormalizationLayer-1/Const_2Const*
_output_shapes
: *
dtype0*
valueB k
(Encoder-2nd-NormalizationLayer-1/Const_3Const*
_output_shapes
: *
dtype0*
valueB �
1Encoder-2nd-NormalizationLayer-1/FusedBatchNormV3FusedBatchNormV31Encoder-2nd-NormalizationLayer-1/Reshape:output:0.Encoder-2nd-NormalizationLayer-1/ones:output:0/Encoder-2nd-NormalizationLayer-1/zeros:output:01Encoder-2nd-NormalizationLayer-1/Const_2:output:01Encoder-2nd-NormalizationLayer-1/Const_3:output:0*
T0*
U0*x
_output_shapesf
d:"������������������:���������:���������:���������:���������:*
data_formatNCHW*
epsilon%o�:�
*Encoder-2nd-NormalizationLayer-1/Reshape_1Reshape5Encoder-2nd-NormalizationLayer-1/FusedBatchNormV3:y:0/Encoder-2nd-NormalizationLayer-1/Shape:output:0*
T0*+
_output_shapes
:���������P	�
3Encoder-2nd-NormalizationLayer-1/mul/ReadVariableOpReadVariableOp<encoder_2nd_normalizationlayer_1_mul_readvariableop_resource*
_output_shapes
:	*
dtype0�
$Encoder-2nd-NormalizationLayer-1/mulMul3Encoder-2nd-NormalizationLayer-1/Reshape_1:output:0;Encoder-2nd-NormalizationLayer-1/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������P	�
3Encoder-2nd-NormalizationLayer-1/add/ReadVariableOpReadVariableOp<encoder_2nd_normalizationlayer_1_add_readvariableop_resource*
_output_shapes
:	*
dtype0�
$Encoder-2nd-NormalizationLayer-1/addAddV2(Encoder-2nd-NormalizationLayer-1/mul:z:0;Encoder-2nd-NormalizationLayer-1/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������P	�
5Encoder-FeedForwardLayer_1_1/Tensordot/ReadVariableOpReadVariableOp>encoder_feedforwardlayer_1_1_tensordot_readvariableop_resource*
_output_shapes

:		*
dtype0u
+Encoder-FeedForwardLayer_1_1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:|
+Encoder-FeedForwardLayer_1_1/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       �
,Encoder-FeedForwardLayer_1_1/Tensordot/ShapeShape(Encoder-2nd-NormalizationLayer-1/add:z:0*
T0*
_output_shapes
::��v
4Encoder-FeedForwardLayer_1_1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
/Encoder-FeedForwardLayer_1_1/Tensordot/GatherV2GatherV25Encoder-FeedForwardLayer_1_1/Tensordot/Shape:output:04Encoder-FeedForwardLayer_1_1/Tensordot/free:output:0=Encoder-FeedForwardLayer_1_1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:x
6Encoder-FeedForwardLayer_1_1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
1Encoder-FeedForwardLayer_1_1/Tensordot/GatherV2_1GatherV25Encoder-FeedForwardLayer_1_1/Tensordot/Shape:output:04Encoder-FeedForwardLayer_1_1/Tensordot/axes:output:0?Encoder-FeedForwardLayer_1_1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:v
,Encoder-FeedForwardLayer_1_1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
+Encoder-FeedForwardLayer_1_1/Tensordot/ProdProd8Encoder-FeedForwardLayer_1_1/Tensordot/GatherV2:output:05Encoder-FeedForwardLayer_1_1/Tensordot/Const:output:0*
T0*
_output_shapes
: x
.Encoder-FeedForwardLayer_1_1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
-Encoder-FeedForwardLayer_1_1/Tensordot/Prod_1Prod:Encoder-FeedForwardLayer_1_1/Tensordot/GatherV2_1:output:07Encoder-FeedForwardLayer_1_1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: t
2Encoder-FeedForwardLayer_1_1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
-Encoder-FeedForwardLayer_1_1/Tensordot/concatConcatV24Encoder-FeedForwardLayer_1_1/Tensordot/free:output:04Encoder-FeedForwardLayer_1_1/Tensordot/axes:output:0;Encoder-FeedForwardLayer_1_1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
,Encoder-FeedForwardLayer_1_1/Tensordot/stackPack4Encoder-FeedForwardLayer_1_1/Tensordot/Prod:output:06Encoder-FeedForwardLayer_1_1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
0Encoder-FeedForwardLayer_1_1/Tensordot/transpose	Transpose(Encoder-2nd-NormalizationLayer-1/add:z:06Encoder-FeedForwardLayer_1_1/Tensordot/concat:output:0*
T0*+
_output_shapes
:���������P	�
.Encoder-FeedForwardLayer_1_1/Tensordot/ReshapeReshape4Encoder-FeedForwardLayer_1_1/Tensordot/transpose:y:05Encoder-FeedForwardLayer_1_1/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
-Encoder-FeedForwardLayer_1_1/Tensordot/MatMulMatMul7Encoder-FeedForwardLayer_1_1/Tensordot/Reshape:output:0=Encoder-FeedForwardLayer_1_1/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������	x
.Encoder-FeedForwardLayer_1_1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:	v
4Encoder-FeedForwardLayer_1_1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
/Encoder-FeedForwardLayer_1_1/Tensordot/concat_1ConcatV28Encoder-FeedForwardLayer_1_1/Tensordot/GatherV2:output:07Encoder-FeedForwardLayer_1_1/Tensordot/Const_2:output:0=Encoder-FeedForwardLayer_1_1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
&Encoder-FeedForwardLayer_1_1/TensordotReshape7Encoder-FeedForwardLayer_1_1/Tensordot/MatMul:product:08Encoder-FeedForwardLayer_1_1/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������P	�
3Encoder-FeedForwardLayer_1_1/BiasAdd/ReadVariableOpReadVariableOp<encoder_feedforwardlayer_1_1_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype0�
$Encoder-FeedForwardLayer_1_1/BiasAddBiasAdd/Encoder-FeedForwardLayer_1_1/Tensordot:output:0;Encoder-FeedForwardLayer_1_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������P	l
'Encoder-FeedForwardLayer_1_1/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
%Encoder-FeedForwardLayer_1_1/Gelu/mulMul0Encoder-FeedForwardLayer_1_1/Gelu/mul/x:output:0-Encoder-FeedForwardLayer_1_1/BiasAdd:output:0*
T0*+
_output_shapes
:���������P	m
(Encoder-FeedForwardLayer_1_1/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *��?�
)Encoder-FeedForwardLayer_1_1/Gelu/truedivRealDiv-Encoder-FeedForwardLayer_1_1/BiasAdd:output:01Encoder-FeedForwardLayer_1_1/Gelu/Cast/x:output:0*
T0*+
_output_shapes
:���������P	�
%Encoder-FeedForwardLayer_1_1/Gelu/ErfErf-Encoder-FeedForwardLayer_1_1/Gelu/truediv:z:0*
T0*+
_output_shapes
:���������P	l
'Encoder-FeedForwardLayer_1_1/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
%Encoder-FeedForwardLayer_1_1/Gelu/addAddV20Encoder-FeedForwardLayer_1_1/Gelu/add/x:output:0)Encoder-FeedForwardLayer_1_1/Gelu/Erf:y:0*
T0*+
_output_shapes
:���������P	�
'Encoder-FeedForwardLayer_1_1/Gelu/mul_1Mul)Encoder-FeedForwardLayer_1_1/Gelu/mul:z:0)Encoder-FeedForwardLayer_1_1/Gelu/add:z:0*
T0*+
_output_shapes
:���������P	�
5Encoder-FeedForwardLayer_2_1/Tensordot/ReadVariableOpReadVariableOp>encoder_feedforwardlayer_2_1_tensordot_readvariableop_resource*
_output_shapes

:		*
dtype0u
+Encoder-FeedForwardLayer_2_1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:|
+Encoder-FeedForwardLayer_2_1/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       �
,Encoder-FeedForwardLayer_2_1/Tensordot/ShapeShape+Encoder-FeedForwardLayer_1_1/Gelu/mul_1:z:0*
T0*
_output_shapes
::��v
4Encoder-FeedForwardLayer_2_1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
/Encoder-FeedForwardLayer_2_1/Tensordot/GatherV2GatherV25Encoder-FeedForwardLayer_2_1/Tensordot/Shape:output:04Encoder-FeedForwardLayer_2_1/Tensordot/free:output:0=Encoder-FeedForwardLayer_2_1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:x
6Encoder-FeedForwardLayer_2_1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
1Encoder-FeedForwardLayer_2_1/Tensordot/GatherV2_1GatherV25Encoder-FeedForwardLayer_2_1/Tensordot/Shape:output:04Encoder-FeedForwardLayer_2_1/Tensordot/axes:output:0?Encoder-FeedForwardLayer_2_1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:v
,Encoder-FeedForwardLayer_2_1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
+Encoder-FeedForwardLayer_2_1/Tensordot/ProdProd8Encoder-FeedForwardLayer_2_1/Tensordot/GatherV2:output:05Encoder-FeedForwardLayer_2_1/Tensordot/Const:output:0*
T0*
_output_shapes
: x
.Encoder-FeedForwardLayer_2_1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
-Encoder-FeedForwardLayer_2_1/Tensordot/Prod_1Prod:Encoder-FeedForwardLayer_2_1/Tensordot/GatherV2_1:output:07Encoder-FeedForwardLayer_2_1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: t
2Encoder-FeedForwardLayer_2_1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
-Encoder-FeedForwardLayer_2_1/Tensordot/concatConcatV24Encoder-FeedForwardLayer_2_1/Tensordot/free:output:04Encoder-FeedForwardLayer_2_1/Tensordot/axes:output:0;Encoder-FeedForwardLayer_2_1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
,Encoder-FeedForwardLayer_2_1/Tensordot/stackPack4Encoder-FeedForwardLayer_2_1/Tensordot/Prod:output:06Encoder-FeedForwardLayer_2_1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
0Encoder-FeedForwardLayer_2_1/Tensordot/transpose	Transpose+Encoder-FeedForwardLayer_1_1/Gelu/mul_1:z:06Encoder-FeedForwardLayer_2_1/Tensordot/concat:output:0*
T0*+
_output_shapes
:���������P	�
.Encoder-FeedForwardLayer_2_1/Tensordot/ReshapeReshape4Encoder-FeedForwardLayer_2_1/Tensordot/transpose:y:05Encoder-FeedForwardLayer_2_1/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
-Encoder-FeedForwardLayer_2_1/Tensordot/MatMulMatMul7Encoder-FeedForwardLayer_2_1/Tensordot/Reshape:output:0=Encoder-FeedForwardLayer_2_1/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������	x
.Encoder-FeedForwardLayer_2_1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:	v
4Encoder-FeedForwardLayer_2_1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
/Encoder-FeedForwardLayer_2_1/Tensordot/concat_1ConcatV28Encoder-FeedForwardLayer_2_1/Tensordot/GatherV2:output:07Encoder-FeedForwardLayer_2_1/Tensordot/Const_2:output:0=Encoder-FeedForwardLayer_2_1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
&Encoder-FeedForwardLayer_2_1/TensordotReshape7Encoder-FeedForwardLayer_2_1/Tensordot/MatMul:product:08Encoder-FeedForwardLayer_2_1/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������P	�
3Encoder-FeedForwardLayer_2_1/BiasAdd/ReadVariableOpReadVariableOp<encoder_feedforwardlayer_2_1_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype0�
$Encoder-FeedForwardLayer_2_1/BiasAddBiasAdd/Encoder-FeedForwardLayer_2_1/Tensordot:output:0;Encoder-FeedForwardLayer_2_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������P	\
dropout_3/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?�
dropout_3/dropout/MulMul-Encoder-FeedForwardLayer_2_1/BiasAdd:output:0 dropout_3/dropout/Const:output:0*
T0*+
_output_shapes
:���������P	�
dropout_3/dropout/ShapeShape-Encoder-FeedForwardLayer_2_1/BiasAdd:output:0*
T0*
_output_shapes
::���
.dropout_3/dropout/random_uniform/RandomUniformRandomUniform dropout_3/dropout/Shape:output:0*
T0*+
_output_shapes
:���������P	*
dtype0*
seed2*
seed��e
 dropout_3/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=�
dropout_3/dropout/GreaterEqualGreaterEqual7dropout_3/dropout/random_uniform/RandomUniform:output:0)dropout_3/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:���������P	^
dropout_3/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout_3/dropout/SelectV2SelectV2"dropout_3/dropout/GreaterEqual:z:0dropout_3/dropout/Mul:z:0"dropout_3/dropout/Const_1:output:0*
T0*+
_output_shapes
:���������P	�
Encoder-2nd-AdditionLayer-1/addAddV2#Encoder-1st-AdditionLayer-1/add:z:0#dropout_3/dropout/SelectV2:output:0*
T0*+
_output_shapes
:���������P	v
IdentityIdentity#Encoder-2nd-AdditionLayer-1/add:z:0^NoOp*
T0*+
_output_shapes
:���������P	�

Identity_1Identity6Encoder-SelfAttentionLayer-1/softmax/Softmax:softmax:0^NoOp*
T0*/
_output_shapes
:���������PP�
NoOpNoOp4^Encoder-1st-NormalizationLayer-1/add/ReadVariableOp4^Encoder-1st-NormalizationLayer-1/mul/ReadVariableOp4^Encoder-2nd-NormalizationLayer-1/add/ReadVariableOp4^Encoder-2nd-NormalizationLayer-1/mul/ReadVariableOp4^Encoder-FeedForwardLayer_1_1/BiasAdd/ReadVariableOp6^Encoder-FeedForwardLayer_1_1/Tensordot/ReadVariableOp4^Encoder-FeedForwardLayer_2_1/BiasAdd/ReadVariableOp6^Encoder-FeedForwardLayer_2_1/Tensordot/ReadVariableOpA^Encoder-SelfAttentionLayer-1/attention_output/add/ReadVariableOpK^Encoder-SelfAttentionLayer-1/attention_output/einsum/Einsum/ReadVariableOp4^Encoder-SelfAttentionLayer-1/key/add/ReadVariableOp>^Encoder-SelfAttentionLayer-1/key/einsum/Einsum/ReadVariableOp6^Encoder-SelfAttentionLayer-1/query/add/ReadVariableOp@^Encoder-SelfAttentionLayer-1/query/einsum/Einsum/ReadVariableOp6^Encoder-SelfAttentionLayer-1/value/add/ReadVariableOp@^Encoder-SelfAttentionLayer-1/value/einsum/Einsum/ReadVariableOp*
_output_shapes
 "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:���������P	: : : : : : : : : : : : : : : : 2j
3Encoder-1st-NormalizationLayer-1/add/ReadVariableOp3Encoder-1st-NormalizationLayer-1/add/ReadVariableOp2j
3Encoder-1st-NormalizationLayer-1/mul/ReadVariableOp3Encoder-1st-NormalizationLayer-1/mul/ReadVariableOp2j
3Encoder-2nd-NormalizationLayer-1/add/ReadVariableOp3Encoder-2nd-NormalizationLayer-1/add/ReadVariableOp2j
3Encoder-2nd-NormalizationLayer-1/mul/ReadVariableOp3Encoder-2nd-NormalizationLayer-1/mul/ReadVariableOp2j
3Encoder-FeedForwardLayer_1_1/BiasAdd/ReadVariableOp3Encoder-FeedForwardLayer_1_1/BiasAdd/ReadVariableOp2n
5Encoder-FeedForwardLayer_1_1/Tensordot/ReadVariableOp5Encoder-FeedForwardLayer_1_1/Tensordot/ReadVariableOp2j
3Encoder-FeedForwardLayer_2_1/BiasAdd/ReadVariableOp3Encoder-FeedForwardLayer_2_1/BiasAdd/ReadVariableOp2n
5Encoder-FeedForwardLayer_2_1/Tensordot/ReadVariableOp5Encoder-FeedForwardLayer_2_1/Tensordot/ReadVariableOp2�
@Encoder-SelfAttentionLayer-1/attention_output/add/ReadVariableOp@Encoder-SelfAttentionLayer-1/attention_output/add/ReadVariableOp2�
JEncoder-SelfAttentionLayer-1/attention_output/einsum/Einsum/ReadVariableOpJEncoder-SelfAttentionLayer-1/attention_output/einsum/Einsum/ReadVariableOp2j
3Encoder-SelfAttentionLayer-1/key/add/ReadVariableOp3Encoder-SelfAttentionLayer-1/key/add/ReadVariableOp2~
=Encoder-SelfAttentionLayer-1/key/einsum/Einsum/ReadVariableOp=Encoder-SelfAttentionLayer-1/key/einsum/Einsum/ReadVariableOp2n
5Encoder-SelfAttentionLayer-1/query/add/ReadVariableOp5Encoder-SelfAttentionLayer-1/query/add/ReadVariableOp2�
?Encoder-SelfAttentionLayer-1/query/einsum/Einsum/ReadVariableOp?Encoder-SelfAttentionLayer-1/query/einsum/Einsum/ReadVariableOp2n
5Encoder-SelfAttentionLayer-1/value/add/ReadVariableOp5Encoder-SelfAttentionLayer-1/value/add/ReadVariableOp2�
?Encoder-SelfAttentionLayer-1/value/einsum/Einsum/ReadVariableOp?Encoder-SelfAttentionLayer-1/value/einsum/Einsum/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:(
$
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:S O
+
_output_shapes
:���������P	
 
_user_specified_nameinputs
�
Q
5__inference_StandardizeTimeLimit_layer_call_fn_233296

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *Y
fTRR
P__inference_StandardizeTimeLimit_layer_call_and_return_conditional_losses_231208`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�+
�
$__inference_signature_wrapper_231873
stacklevelinputfeatures
timelimitinput
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:
	unknown_3:	
	unknown_4:
	unknown_5:	
	unknown_6:
	unknown_7:	
	unknown_8:	
	unknown_9:	

unknown_10:	

unknown_11:		

unknown_12:	

unknown_13:		

unknown_14:	

unknown_15:	

unknown_16:	 

unknown_17:	

unknown_18: 

unknown_19:	

unknown_20: 

unknown_21:	

unknown_22: 

unknown_23:	

unknown_24:	

unknown_25:	

unknown_26:	

unknown_27:		

unknown_28:	

unknown_29:		

unknown_30:	

unknown_31:	

unknown_32:	 

unknown_33:	

unknown_34: 

unknown_35:	

unknown_36: 

unknown_37:	

unknown_38: 

unknown_39:	

unknown_40:	

unknown_41:	

unknown_42:	

unknown_43:		

unknown_44:	

unknown_45:		

unknown_46:	

unknown_47:	

unknown_48:	

unknown_49:



unknown_50:


unknown_51:


unknown_52:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallstacklevelinputfeaturestimelimitinputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50
unknown_51
unknown_52*C
Tin<
:28*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:*X
_read_only_resource_inputs:
86	
 !"#$%&'()*+,-./01234567*0
config_proto 

CPU

GPU2*0J 8� **
f%R#
!__inference__wrapped_model_229753`
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes
:<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:���������P	:���������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&7"
 
_user_specified_name231869:&6"
 
_user_specified_name231867:&5"
 
_user_specified_name231865:&4"
 
_user_specified_name231863:&3"
 
_user_specified_name231861:&2"
 
_user_specified_name231859:&1"
 
_user_specified_name231857:&0"
 
_user_specified_name231855:&/"
 
_user_specified_name231853:&."
 
_user_specified_name231851:&-"
 
_user_specified_name231849:&,"
 
_user_specified_name231847:&+"
 
_user_specified_name231845:&*"
 
_user_specified_name231843:&)"
 
_user_specified_name231841:&("
 
_user_specified_name231839:&'"
 
_user_specified_name231837:&&"
 
_user_specified_name231835:&%"
 
_user_specified_name231833:&$"
 
_user_specified_name231831:&#"
 
_user_specified_name231829:&""
 
_user_specified_name231827:&!"
 
_user_specified_name231825:& "
 
_user_specified_name231823:&"
 
_user_specified_name231821:&"
 
_user_specified_name231819:&"
 
_user_specified_name231817:&"
 
_user_specified_name231815:&"
 
_user_specified_name231813:&"
 
_user_specified_name231811:&"
 
_user_specified_name231809:&"
 
_user_specified_name231807:&"
 
_user_specified_name231805:&"
 
_user_specified_name231803:&"
 
_user_specified_name231801:&"
 
_user_specified_name231799:&"
 
_user_specified_name231797:&"
 
_user_specified_name231795:&"
 
_user_specified_name231793:&"
 
_user_specified_name231791:&"
 
_user_specified_name231789:&"
 
_user_specified_name231787:&"
 
_user_specified_name231785:&"
 
_user_specified_name231783:&"
 
_user_specified_name231781:&
"
 
_user_specified_name231779:&	"
 
_user_specified_name231777:&"
 
_user_specified_name231775:&"
 
_user_specified_name231773:&"
 
_user_specified_name231771:&"
 
_user_specified_name231769:&"
 
_user_specified_name231767:&"
 
_user_specified_name231765:&"
 
_user_specified_name231763:WS
'
_output_shapes
:���������
(
_user_specified_nameTimeLimitInput:d `
+
_output_shapes
:���������P	
1
_user_specified_nameStackLevelInputFeatures
�
I
-__inference_MaskingLayer_layer_call_fn_231878

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������P	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_MaskingLayer_layer_call_and_return_conditional_losses_229767d
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:���������P	"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������P	:S O
+
_output_shapes
:���������P	
 
_user_specified_nameinputs
�e
�
C__inference_model_1_layer_call_and_return_conditional_losses_230556
stacklevelinputfeatures
timelimitinput*
transformer_encoder_3_229958:	*
transformer_encoder_3_229960:	2
transformer_encoder_3_229962:	.
transformer_encoder_3_229964:2
transformer_encoder_3_229966:	.
transformer_encoder_3_229968:2
transformer_encoder_3_229970:	.
transformer_encoder_3_229972:2
transformer_encoder_3_229974:	*
transformer_encoder_3_229976:	*
transformer_encoder_3_229978:	*
transformer_encoder_3_229980:	.
transformer_encoder_3_229982:		*
transformer_encoder_3_229984:	.
transformer_encoder_3_229986:		*
transformer_encoder_3_229988:	*
transformer_encoder_4_230180:	*
transformer_encoder_4_230182:	2
transformer_encoder_4_230184:	.
transformer_encoder_4_230186:2
transformer_encoder_4_230188:	.
transformer_encoder_4_230190:2
transformer_encoder_4_230192:	.
transformer_encoder_4_230194:2
transformer_encoder_4_230196:	*
transformer_encoder_4_230198:	*
transformer_encoder_4_230200:	*
transformer_encoder_4_230202:	.
transformer_encoder_4_230204:		*
transformer_encoder_4_230206:	.
transformer_encoder_4_230208:		*
transformer_encoder_4_230210:	*
transformer_encoder_5_230402:	*
transformer_encoder_5_230404:	2
transformer_encoder_5_230406:	.
transformer_encoder_5_230408:2
transformer_encoder_5_230410:	.
transformer_encoder_5_230412:2
transformer_encoder_5_230414:	.
transformer_encoder_5_230416:2
transformer_encoder_5_230418:	*
transformer_encoder_5_230420:	*
transformer_encoder_5_230422:	*
transformer_encoder_5_230424:	.
transformer_encoder_5_230426:		*
transformer_encoder_5_230428:	.
transformer_encoder_5_230430:		*
transformer_encoder_5_230432:	#
finallayernorm_230478:	#
finallayernorm_230480:	7
%fullyconnectedlayerimprovement_230528:

3
%fullyconnectedlayerimprovement_230530:
.
predictionimprovement_230544:
*
predictionimprovement_230546:
identity��&FinalLayerNorm/StatefulPartitionedCall�6FullyConnectedLayerImprovement/StatefulPartitionedCall�-PredictionImprovement/StatefulPartitionedCall�-transformer_encoder_3/StatefulPartitionedCall�-transformer_encoder_4/StatefulPartitionedCall�-transformer_encoder_5/StatefulPartitionedCall�
MaskingLayer/PartitionedCallPartitionedCallstacklevelinputfeatures*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������P	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_MaskingLayer_layer_call_and_return_conditional_losses_229767�
transformer_encoder_3/CastCast%MaskingLayer/PartitionedCall:output:0*

DstT0*

SrcT0*+
_output_shapes
:���������P	�
-transformer_encoder_3/StatefulPartitionedCallStatefulPartitionedCalltransformer_encoder_3/Cast:y:0transformer_encoder_3_229958transformer_encoder_3_229960transformer_encoder_3_229962transformer_encoder_3_229964transformer_encoder_3_229966transformer_encoder_3_229968transformer_encoder_3_229970transformer_encoder_3_229972transformer_encoder_3_229974transformer_encoder_3_229976transformer_encoder_3_229978transformer_encoder_3_229980transformer_encoder_3_229982transformer_encoder_3_229984transformer_encoder_3_229986transformer_encoder_3_229988*
Tin
2*
Tout
2*
_collective_manager_ids
 *F
_output_shapes4
2:���������P	:���������PP*2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *Z
fURS
Q__inference_transformer_encoder_3_layer_call_and_return_conditional_losses_229957�
-transformer_encoder_4/StatefulPartitionedCallStatefulPartitionedCall6transformer_encoder_3/StatefulPartitionedCall:output:0transformer_encoder_4_230180transformer_encoder_4_230182transformer_encoder_4_230184transformer_encoder_4_230186transformer_encoder_4_230188transformer_encoder_4_230190transformer_encoder_4_230192transformer_encoder_4_230194transformer_encoder_4_230196transformer_encoder_4_230198transformer_encoder_4_230200transformer_encoder_4_230202transformer_encoder_4_230204transformer_encoder_4_230206transformer_encoder_4_230208transformer_encoder_4_230210*
Tin
2*
Tout
2*
_collective_manager_ids
 *F
_output_shapes4
2:���������P	:���������PP*2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *Z
fURS
Q__inference_transformer_encoder_4_layer_call_and_return_conditional_losses_230179�
-transformer_encoder_5/StatefulPartitionedCallStatefulPartitionedCall6transformer_encoder_4/StatefulPartitionedCall:output:0transformer_encoder_5_230402transformer_encoder_5_230404transformer_encoder_5_230406transformer_encoder_5_230408transformer_encoder_5_230410transformer_encoder_5_230412transformer_encoder_5_230414transformer_encoder_5_230416transformer_encoder_5_230418transformer_encoder_5_230420transformer_encoder_5_230422transformer_encoder_5_230424transformer_encoder_5_230426transformer_encoder_5_230428transformer_encoder_5_230430transformer_encoder_5_230432*
Tin
2*
Tout
2*
_collective_manager_ids
 *F
_output_shapes4
2:���������P	:���������PP*2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *Z
fURS
Q__inference_transformer_encoder_5_layer_call_and_return_conditional_losses_230401�
&FinalLayerNorm/StatefulPartitionedCallStatefulPartitionedCall6transformer_encoder_5/StatefulPartitionedCall:output:0finallayernorm_230478finallayernorm_230480*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������P	*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_FinalLayerNorm_layer_call_and_return_conditional_losses_230477�
0ReduceStackDimensionViaSummation/PartitionedCallPartitionedCall/FinalLayerNorm/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *e
f`R^
\__inference_ReduceStackDimensionViaSummation_layer_call_and_return_conditional_losses_230490r
StandardizeTimeLimit/CastCasttimelimitinput*

DstT0*

SrcT0*'
_output_shapes
:����������
$StandardizeTimeLimit/PartitionedCallPartitionedCallStandardizeTimeLimit/Cast:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *Y
fTRR
P__inference_StandardizeTimeLimit_layer_call_and_return_conditional_losses_230500�
 ConcatenateLayer/PartitionedCallPartitionedCall9ReduceStackDimensionViaSummation/PartitionedCall:output:0-StandardizeTimeLimit/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_ConcatenateLayer_layer_call_and_return_conditional_losses_230508�
6FullyConnectedLayerImprovement/StatefulPartitionedCallStatefulPartitionedCall)ConcatenateLayer/PartitionedCall:output:0%fullyconnectedlayerimprovement_230528%fullyconnectedlayerimprovement_230530*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *c
f^R\
Z__inference_FullyConnectedLayerImprovement_layer_call_and_return_conditional_losses_230527�
-PredictionImprovement/StatefulPartitionedCallStatefulPartitionedCall?FullyConnectedLayerImprovement/StatefulPartitionedCall:output:0predictionimprovement_230544predictionimprovement_230546*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Z
fURS
Q__inference_PredictionImprovement_layer_call_and_return_conditional_losses_230543�
Output/PartitionedCallPartitionedCall6PredictionImprovement/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_Output_layer_call_and_return_conditional_losses_230553_
IdentityIdentityOutput/PartitionedCall:output:0^NoOp*
T0*
_output_shapes
:�
NoOpNoOp'^FinalLayerNorm/StatefulPartitionedCall7^FullyConnectedLayerImprovement/StatefulPartitionedCall.^PredictionImprovement/StatefulPartitionedCall.^transformer_encoder_3/StatefulPartitionedCall.^transformer_encoder_4/StatefulPartitionedCall.^transformer_encoder_5/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:���������P	:���������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2P
&FinalLayerNorm/StatefulPartitionedCall&FinalLayerNorm/StatefulPartitionedCall2p
6FullyConnectedLayerImprovement/StatefulPartitionedCall6FullyConnectedLayerImprovement/StatefulPartitionedCall2^
-PredictionImprovement/StatefulPartitionedCall-PredictionImprovement/StatefulPartitionedCall2^
-transformer_encoder_3/StatefulPartitionedCall-transformer_encoder_3/StatefulPartitionedCall2^
-transformer_encoder_4/StatefulPartitionedCall-transformer_encoder_4/StatefulPartitionedCall2^
-transformer_encoder_5/StatefulPartitionedCall-transformer_encoder_5/StatefulPartitionedCall:&7"
 
_user_specified_name230546:&6"
 
_user_specified_name230544:&5"
 
_user_specified_name230530:&4"
 
_user_specified_name230528:&3"
 
_user_specified_name230480:&2"
 
_user_specified_name230478:&1"
 
_user_specified_name230432:&0"
 
_user_specified_name230430:&/"
 
_user_specified_name230428:&."
 
_user_specified_name230426:&-"
 
_user_specified_name230424:&,"
 
_user_specified_name230422:&+"
 
_user_specified_name230420:&*"
 
_user_specified_name230418:&)"
 
_user_specified_name230416:&("
 
_user_specified_name230414:&'"
 
_user_specified_name230412:&&"
 
_user_specified_name230410:&%"
 
_user_specified_name230408:&$"
 
_user_specified_name230406:&#"
 
_user_specified_name230404:&""
 
_user_specified_name230402:&!"
 
_user_specified_name230210:& "
 
_user_specified_name230208:&"
 
_user_specified_name230206:&"
 
_user_specified_name230204:&"
 
_user_specified_name230202:&"
 
_user_specified_name230200:&"
 
_user_specified_name230198:&"
 
_user_specified_name230196:&"
 
_user_specified_name230194:&"
 
_user_specified_name230192:&"
 
_user_specified_name230190:&"
 
_user_specified_name230188:&"
 
_user_specified_name230186:&"
 
_user_specified_name230184:&"
 
_user_specified_name230182:&"
 
_user_specified_name230180:&"
 
_user_specified_name229988:&"
 
_user_specified_name229986:&"
 
_user_specified_name229984:&"
 
_user_specified_name229982:&"
 
_user_specified_name229980:&"
 
_user_specified_name229978:&"
 
_user_specified_name229976:&
"
 
_user_specified_name229974:&	"
 
_user_specified_name229972:&"
 
_user_specified_name229970:&"
 
_user_specified_name229968:&"
 
_user_specified_name229966:&"
 
_user_specified_name229964:&"
 
_user_specified_name229962:&"
 
_user_specified_name229960:&"
 
_user_specified_name229958:WS
'
_output_shapes
:���������
(
_user_specified_nameTimeLimitInput:d `
+
_output_shapes
:���������P	
1
_user_specified_nameStackLevelInputFeatures
�+
�
(__inference_model_1_layer_call_fn_231342
stacklevelinputfeatures
timelimitinput
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:
	unknown_3:	
	unknown_4:
	unknown_5:	
	unknown_6:
	unknown_7:	
	unknown_8:	
	unknown_9:	

unknown_10:	

unknown_11:		

unknown_12:	

unknown_13:		

unknown_14:	

unknown_15:	

unknown_16:	 

unknown_17:	

unknown_18: 

unknown_19:	

unknown_20: 

unknown_21:	

unknown_22: 

unknown_23:	

unknown_24:	

unknown_25:	

unknown_26:	

unknown_27:		

unknown_28:	

unknown_29:		

unknown_30:	

unknown_31:	

unknown_32:	 

unknown_33:	

unknown_34: 

unknown_35:	

unknown_36: 

unknown_37:	

unknown_38: 

unknown_39:	

unknown_40:	

unknown_41:	

unknown_42:	

unknown_43:		

unknown_44:	

unknown_45:		

unknown_46:	

unknown_47:	

unknown_48:	

unknown_49:



unknown_50:


unknown_51:


unknown_52:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallstacklevelinputfeaturestimelimitinputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50
unknown_51
unknown_52*C
Tin<
:28*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:*X
_read_only_resource_inputs:
86	
 !"#$%&'()*+,-./01234567*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_model_1_layer_call_and_return_conditional_losses_230556`
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes
:<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:���������P	:���������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&7"
 
_user_specified_name231338:&6"
 
_user_specified_name231336:&5"
 
_user_specified_name231334:&4"
 
_user_specified_name231332:&3"
 
_user_specified_name231330:&2"
 
_user_specified_name231328:&1"
 
_user_specified_name231326:&0"
 
_user_specified_name231324:&/"
 
_user_specified_name231322:&."
 
_user_specified_name231320:&-"
 
_user_specified_name231318:&,"
 
_user_specified_name231316:&+"
 
_user_specified_name231314:&*"
 
_user_specified_name231312:&)"
 
_user_specified_name231310:&("
 
_user_specified_name231308:&'"
 
_user_specified_name231306:&&"
 
_user_specified_name231304:&%"
 
_user_specified_name231302:&$"
 
_user_specified_name231300:&#"
 
_user_specified_name231298:&""
 
_user_specified_name231296:&!"
 
_user_specified_name231294:& "
 
_user_specified_name231292:&"
 
_user_specified_name231290:&"
 
_user_specified_name231288:&"
 
_user_specified_name231286:&"
 
_user_specified_name231284:&"
 
_user_specified_name231282:&"
 
_user_specified_name231280:&"
 
_user_specified_name231278:&"
 
_user_specified_name231276:&"
 
_user_specified_name231274:&"
 
_user_specified_name231272:&"
 
_user_specified_name231270:&"
 
_user_specified_name231268:&"
 
_user_specified_name231266:&"
 
_user_specified_name231264:&"
 
_user_specified_name231262:&"
 
_user_specified_name231260:&"
 
_user_specified_name231258:&"
 
_user_specified_name231256:&"
 
_user_specified_name231254:&"
 
_user_specified_name231252:&"
 
_user_specified_name231250:&
"
 
_user_specified_name231248:&	"
 
_user_specified_name231246:&"
 
_user_specified_name231244:&"
 
_user_specified_name231242:&"
 
_user_specified_name231240:&"
 
_user_specified_name231238:&"
 
_user_specified_name231236:&"
 
_user_specified_name231234:&"
 
_user_specified_name231232:WS
'
_output_shapes
:���������
(
_user_specified_nameTimeLimitInput:d `
+
_output_shapes
:���������P	
1
_user_specified_nameStackLevelInputFeatures
�
x
L__inference_ConcatenateLayer_layer_call_and_return_conditional_losses_233325
inputs_0
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :w
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*'
_output_shapes
:���������
W
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:���������
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:���������	:���������:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs_1:Q M
'
_output_shapes
:���������	
"
_user_specified_name
inputs_0
�
v
L__inference_ConcatenateLayer_layer_call_and_return_conditional_losses_230508

inputs
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :u
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*'
_output_shapes
:���������
W
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:���������
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:���������	:���������:OK
'
_output_shapes
:���������
 
_user_specified_nameinputs:O K
'
_output_shapes
:���������	
 
_user_specified_nameinputs
��
�
Q__inference_transformer_encoder_5_layer_call_and_return_conditional_losses_233209

inputsJ
<encoder_1st_normalizationlayer_3_mul_readvariableop_resource:	J
<encoder_1st_normalizationlayer_3_add_readvariableop_resource:	^
Hencoder_selfattentionlayer_3_query_einsum_einsum_readvariableop_resource:	P
>encoder_selfattentionlayer_3_query_add_readvariableop_resource:\
Fencoder_selfattentionlayer_3_key_einsum_einsum_readvariableop_resource:	N
<encoder_selfattentionlayer_3_key_add_readvariableop_resource:^
Hencoder_selfattentionlayer_3_value_einsum_einsum_readvariableop_resource:	P
>encoder_selfattentionlayer_3_value_add_readvariableop_resource:i
Sencoder_selfattentionlayer_3_attention_output_einsum_einsum_readvariableop_resource:	W
Iencoder_selfattentionlayer_3_attention_output_add_readvariableop_resource:	J
<encoder_2nd_normalizationlayer_3_mul_readvariableop_resource:	J
<encoder_2nd_normalizationlayer_3_add_readvariableop_resource:	P
>encoder_feedforwardlayer_1_3_tensordot_readvariableop_resource:		J
<encoder_feedforwardlayer_1_3_biasadd_readvariableop_resource:	P
>encoder_feedforwardlayer_2_3_tensordot_readvariableop_resource:		J
<encoder_feedforwardlayer_2_3_biasadd_readvariableop_resource:	
identity

identity_1��3Encoder-1st-NormalizationLayer-3/add/ReadVariableOp�3Encoder-1st-NormalizationLayer-3/mul/ReadVariableOp�3Encoder-2nd-NormalizationLayer-3/add/ReadVariableOp�3Encoder-2nd-NormalizationLayer-3/mul/ReadVariableOp�3Encoder-FeedForwardLayer_1_3/BiasAdd/ReadVariableOp�5Encoder-FeedForwardLayer_1_3/Tensordot/ReadVariableOp�3Encoder-FeedForwardLayer_2_3/BiasAdd/ReadVariableOp�5Encoder-FeedForwardLayer_2_3/Tensordot/ReadVariableOp�@Encoder-SelfAttentionLayer-3/attention_output/add/ReadVariableOp�JEncoder-SelfAttentionLayer-3/attention_output/einsum/Einsum/ReadVariableOp�3Encoder-SelfAttentionLayer-3/key/add/ReadVariableOp�=Encoder-SelfAttentionLayer-3/key/einsum/Einsum/ReadVariableOp�5Encoder-SelfAttentionLayer-3/query/add/ReadVariableOp�?Encoder-SelfAttentionLayer-3/query/einsum/Einsum/ReadVariableOp�5Encoder-SelfAttentionLayer-3/value/add/ReadVariableOp�?Encoder-SelfAttentionLayer-3/value/einsum/Einsum/ReadVariableOpj
&Encoder-1st-NormalizationLayer-3/ShapeShapeinputs*
T0*
_output_shapes
::��~
4Encoder-1st-NormalizationLayer-3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
6Encoder-1st-NormalizationLayer-3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
6Encoder-1st-NormalizationLayer-3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
.Encoder-1st-NormalizationLayer-3/strided_sliceStridedSlice/Encoder-1st-NormalizationLayer-3/Shape:output:0=Encoder-1st-NormalizationLayer-3/strided_slice/stack:output:0?Encoder-1st-NormalizationLayer-3/strided_slice/stack_1:output:0?Encoder-1st-NormalizationLayer-3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskp
&Encoder-1st-NormalizationLayer-3/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
%Encoder-1st-NormalizationLayer-3/ProdProd7Encoder-1st-NormalizationLayer-3/strided_slice:output:0/Encoder-1st-NormalizationLayer-3/Const:output:0*
T0*
_output_shapes
: �
6Encoder-1st-NormalizationLayer-3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:�
8Encoder-1st-NormalizationLayer-3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: �
8Encoder-1st-NormalizationLayer-3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
0Encoder-1st-NormalizationLayer-3/strided_slice_1StridedSlice/Encoder-1st-NormalizationLayer-3/Shape:output:0?Encoder-1st-NormalizationLayer-3/strided_slice_1/stack:output:0AEncoder-1st-NormalizationLayer-3/strided_slice_1/stack_1:output:0AEncoder-1st-NormalizationLayer-3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_maskr
(Encoder-1st-NormalizationLayer-3/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
'Encoder-1st-NormalizationLayer-3/Prod_1Prod9Encoder-1st-NormalizationLayer-3/strided_slice_1:output:01Encoder-1st-NormalizationLayer-3/Const_1:output:0*
T0*
_output_shapes
: r
0Encoder-1st-NormalizationLayer-3/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :r
0Encoder-1st-NormalizationLayer-3/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :�
.Encoder-1st-NormalizationLayer-3/Reshape/shapePack9Encoder-1st-NormalizationLayer-3/Reshape/shape/0:output:0.Encoder-1st-NormalizationLayer-3/Prod:output:00Encoder-1st-NormalizationLayer-3/Prod_1:output:09Encoder-1st-NormalizationLayer-3/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:�
(Encoder-1st-NormalizationLayer-3/ReshapeReshapeinputs7Encoder-1st-NormalizationLayer-3/Reshape/shape:output:0*
T0*8
_output_shapes&
$:"�������������������
,Encoder-1st-NormalizationLayer-3/ones/packedPack.Encoder-1st-NormalizationLayer-3/Prod:output:0*
N*
T0*
_output_shapes
:p
+Encoder-1st-NormalizationLayer-3/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
%Encoder-1st-NormalizationLayer-3/onesFill5Encoder-1st-NormalizationLayer-3/ones/packed:output:04Encoder-1st-NormalizationLayer-3/ones/Const:output:0*
T0*#
_output_shapes
:����������
-Encoder-1st-NormalizationLayer-3/zeros/packedPack.Encoder-1st-NormalizationLayer-3/Prod:output:0*
N*
T0*
_output_shapes
:q
,Encoder-1st-NormalizationLayer-3/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
&Encoder-1st-NormalizationLayer-3/zerosFill6Encoder-1st-NormalizationLayer-3/zeros/packed:output:05Encoder-1st-NormalizationLayer-3/zeros/Const:output:0*
T0*#
_output_shapes
:���������k
(Encoder-1st-NormalizationLayer-3/Const_2Const*
_output_shapes
: *
dtype0*
valueB k
(Encoder-1st-NormalizationLayer-3/Const_3Const*
_output_shapes
: *
dtype0*
valueB �
1Encoder-1st-NormalizationLayer-3/FusedBatchNormV3FusedBatchNormV31Encoder-1st-NormalizationLayer-3/Reshape:output:0.Encoder-1st-NormalizationLayer-3/ones:output:0/Encoder-1st-NormalizationLayer-3/zeros:output:01Encoder-1st-NormalizationLayer-3/Const_2:output:01Encoder-1st-NormalizationLayer-3/Const_3:output:0*
T0*
U0*x
_output_shapesf
d:"������������������:���������:���������:���������:���������:*
data_formatNCHW*
epsilon%o�:�
*Encoder-1st-NormalizationLayer-3/Reshape_1Reshape5Encoder-1st-NormalizationLayer-3/FusedBatchNormV3:y:0/Encoder-1st-NormalizationLayer-3/Shape:output:0*
T0*+
_output_shapes
:���������P	�
3Encoder-1st-NormalizationLayer-3/mul/ReadVariableOpReadVariableOp<encoder_1st_normalizationlayer_3_mul_readvariableop_resource*
_output_shapes
:	*
dtype0�
$Encoder-1st-NormalizationLayer-3/mulMul3Encoder-1st-NormalizationLayer-3/Reshape_1:output:0;Encoder-1st-NormalizationLayer-3/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������P	�
3Encoder-1st-NormalizationLayer-3/add/ReadVariableOpReadVariableOp<encoder_1st_normalizationlayer_3_add_readvariableop_resource*
_output_shapes
:	*
dtype0�
$Encoder-1st-NormalizationLayer-3/addAddV2(Encoder-1st-NormalizationLayer-3/mul:z:0;Encoder-1st-NormalizationLayer-3/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������P	�
?Encoder-SelfAttentionLayer-3/query/einsum/Einsum/ReadVariableOpReadVariableOpHencoder_selfattentionlayer_3_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:	*
dtype0�
0Encoder-SelfAttentionLayer-3/query/einsum/EinsumEinsum(Encoder-1st-NormalizationLayer-3/add:z:0GEncoder-SelfAttentionLayer-3/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������P*
equationabc,cde->abde�
5Encoder-SelfAttentionLayer-3/query/add/ReadVariableOpReadVariableOp>encoder_selfattentionlayer_3_query_add_readvariableop_resource*
_output_shapes

:*
dtype0�
&Encoder-SelfAttentionLayer-3/query/addAddV29Encoder-SelfAttentionLayer-3/query/einsum/Einsum:output:0=Encoder-SelfAttentionLayer-3/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������P�
=Encoder-SelfAttentionLayer-3/key/einsum/Einsum/ReadVariableOpReadVariableOpFencoder_selfattentionlayer_3_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:	*
dtype0�
.Encoder-SelfAttentionLayer-3/key/einsum/EinsumEinsum(Encoder-1st-NormalizationLayer-3/add:z:0EEncoder-SelfAttentionLayer-3/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������P*
equationabc,cde->abde�
3Encoder-SelfAttentionLayer-3/key/add/ReadVariableOpReadVariableOp<encoder_selfattentionlayer_3_key_add_readvariableop_resource*
_output_shapes

:*
dtype0�
$Encoder-SelfAttentionLayer-3/key/addAddV27Encoder-SelfAttentionLayer-3/key/einsum/Einsum:output:0;Encoder-SelfAttentionLayer-3/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������P�
?Encoder-SelfAttentionLayer-3/value/einsum/Einsum/ReadVariableOpReadVariableOpHencoder_selfattentionlayer_3_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:	*
dtype0�
0Encoder-SelfAttentionLayer-3/value/einsum/EinsumEinsum(Encoder-1st-NormalizationLayer-3/add:z:0GEncoder-SelfAttentionLayer-3/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������P*
equationabc,cde->abde�
5Encoder-SelfAttentionLayer-3/value/add/ReadVariableOpReadVariableOp>encoder_selfattentionlayer_3_value_add_readvariableop_resource*
_output_shapes

:*
dtype0�
&Encoder-SelfAttentionLayer-3/value/addAddV29Encoder-SelfAttentionLayer-3/value/einsum/Einsum:output:0=Encoder-SelfAttentionLayer-3/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������Pg
"Encoder-SelfAttentionLayer-3/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *:�?�
 Encoder-SelfAttentionLayer-3/MulMul*Encoder-SelfAttentionLayer-3/query/add:z:0+Encoder-SelfAttentionLayer-3/Mul/y:output:0*
T0*/
_output_shapes
:���������P�
*Encoder-SelfAttentionLayer-3/einsum/EinsumEinsum(Encoder-SelfAttentionLayer-3/key/add:z:0$Encoder-SelfAttentionLayer-3/Mul:z:0*
N*
T0*/
_output_shapes
:���������PP*
equationaecd,abcd->acbe�
,Encoder-SelfAttentionLayer-3/softmax/SoftmaxSoftmax3Encoder-SelfAttentionLayer-3/einsum/Einsum:output:0*
T0*/
_output_shapes
:���������PP�
-Encoder-SelfAttentionLayer-3/dropout/IdentityIdentity6Encoder-SelfAttentionLayer-3/softmax/Softmax:softmax:0*
T0*/
_output_shapes
:���������PP�
,Encoder-SelfAttentionLayer-3/einsum_1/EinsumEinsum6Encoder-SelfAttentionLayer-3/dropout/Identity:output:0*Encoder-SelfAttentionLayer-3/value/add:z:0*
N*
T0*/
_output_shapes
:���������P*
equationacbe,aecd->abcd�
JEncoder-SelfAttentionLayer-3/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpSencoder_selfattentionlayer_3_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:	*
dtype0�
;Encoder-SelfAttentionLayer-3/attention_output/einsum/EinsumEinsum5Encoder-SelfAttentionLayer-3/einsum_1/Einsum:output:0REncoder-SelfAttentionLayer-3/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:���������P	*
equationabcd,cde->abe�
@Encoder-SelfAttentionLayer-3/attention_output/add/ReadVariableOpReadVariableOpIencoder_selfattentionlayer_3_attention_output_add_readvariableop_resource*
_output_shapes
:	*
dtype0�
1Encoder-SelfAttentionLayer-3/attention_output/addAddV2DEncoder-SelfAttentionLayer-3/attention_output/einsum/Einsum:output:0HEncoder-SelfAttentionLayer-3/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������P	�
Encoder-1st-AdditionLayer-3/addAddV2inputs5Encoder-SelfAttentionLayer-3/attention_output/add:z:0*
T0*+
_output_shapes
:���������P	�
&Encoder-2nd-NormalizationLayer-3/ShapeShape#Encoder-1st-AdditionLayer-3/add:z:0*
T0*
_output_shapes
::��~
4Encoder-2nd-NormalizationLayer-3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
6Encoder-2nd-NormalizationLayer-3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
6Encoder-2nd-NormalizationLayer-3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
.Encoder-2nd-NormalizationLayer-3/strided_sliceStridedSlice/Encoder-2nd-NormalizationLayer-3/Shape:output:0=Encoder-2nd-NormalizationLayer-3/strided_slice/stack:output:0?Encoder-2nd-NormalizationLayer-3/strided_slice/stack_1:output:0?Encoder-2nd-NormalizationLayer-3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskp
&Encoder-2nd-NormalizationLayer-3/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
%Encoder-2nd-NormalizationLayer-3/ProdProd7Encoder-2nd-NormalizationLayer-3/strided_slice:output:0/Encoder-2nd-NormalizationLayer-3/Const:output:0*
T0*
_output_shapes
: �
6Encoder-2nd-NormalizationLayer-3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:�
8Encoder-2nd-NormalizationLayer-3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: �
8Encoder-2nd-NormalizationLayer-3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
0Encoder-2nd-NormalizationLayer-3/strided_slice_1StridedSlice/Encoder-2nd-NormalizationLayer-3/Shape:output:0?Encoder-2nd-NormalizationLayer-3/strided_slice_1/stack:output:0AEncoder-2nd-NormalizationLayer-3/strided_slice_1/stack_1:output:0AEncoder-2nd-NormalizationLayer-3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_maskr
(Encoder-2nd-NormalizationLayer-3/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
'Encoder-2nd-NormalizationLayer-3/Prod_1Prod9Encoder-2nd-NormalizationLayer-3/strided_slice_1:output:01Encoder-2nd-NormalizationLayer-3/Const_1:output:0*
T0*
_output_shapes
: r
0Encoder-2nd-NormalizationLayer-3/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :r
0Encoder-2nd-NormalizationLayer-3/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :�
.Encoder-2nd-NormalizationLayer-3/Reshape/shapePack9Encoder-2nd-NormalizationLayer-3/Reshape/shape/0:output:0.Encoder-2nd-NormalizationLayer-3/Prod:output:00Encoder-2nd-NormalizationLayer-3/Prod_1:output:09Encoder-2nd-NormalizationLayer-3/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:�
(Encoder-2nd-NormalizationLayer-3/ReshapeReshape#Encoder-1st-AdditionLayer-3/add:z:07Encoder-2nd-NormalizationLayer-3/Reshape/shape:output:0*
T0*8
_output_shapes&
$:"�������������������
,Encoder-2nd-NormalizationLayer-3/ones/packedPack.Encoder-2nd-NormalizationLayer-3/Prod:output:0*
N*
T0*
_output_shapes
:p
+Encoder-2nd-NormalizationLayer-3/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
%Encoder-2nd-NormalizationLayer-3/onesFill5Encoder-2nd-NormalizationLayer-3/ones/packed:output:04Encoder-2nd-NormalizationLayer-3/ones/Const:output:0*
T0*#
_output_shapes
:����������
-Encoder-2nd-NormalizationLayer-3/zeros/packedPack.Encoder-2nd-NormalizationLayer-3/Prod:output:0*
N*
T0*
_output_shapes
:q
,Encoder-2nd-NormalizationLayer-3/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
&Encoder-2nd-NormalizationLayer-3/zerosFill6Encoder-2nd-NormalizationLayer-3/zeros/packed:output:05Encoder-2nd-NormalizationLayer-3/zeros/Const:output:0*
T0*#
_output_shapes
:���������k
(Encoder-2nd-NormalizationLayer-3/Const_2Const*
_output_shapes
: *
dtype0*
valueB k
(Encoder-2nd-NormalizationLayer-3/Const_3Const*
_output_shapes
: *
dtype0*
valueB �
1Encoder-2nd-NormalizationLayer-3/FusedBatchNormV3FusedBatchNormV31Encoder-2nd-NormalizationLayer-3/Reshape:output:0.Encoder-2nd-NormalizationLayer-3/ones:output:0/Encoder-2nd-NormalizationLayer-3/zeros:output:01Encoder-2nd-NormalizationLayer-3/Const_2:output:01Encoder-2nd-NormalizationLayer-3/Const_3:output:0*
T0*
U0*x
_output_shapesf
d:"������������������:���������:���������:���������:���������:*
data_formatNCHW*
epsilon%o�:�
*Encoder-2nd-NormalizationLayer-3/Reshape_1Reshape5Encoder-2nd-NormalizationLayer-3/FusedBatchNormV3:y:0/Encoder-2nd-NormalizationLayer-3/Shape:output:0*
T0*+
_output_shapes
:���������P	�
3Encoder-2nd-NormalizationLayer-3/mul/ReadVariableOpReadVariableOp<encoder_2nd_normalizationlayer_3_mul_readvariableop_resource*
_output_shapes
:	*
dtype0�
$Encoder-2nd-NormalizationLayer-3/mulMul3Encoder-2nd-NormalizationLayer-3/Reshape_1:output:0;Encoder-2nd-NormalizationLayer-3/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������P	�
3Encoder-2nd-NormalizationLayer-3/add/ReadVariableOpReadVariableOp<encoder_2nd_normalizationlayer_3_add_readvariableop_resource*
_output_shapes
:	*
dtype0�
$Encoder-2nd-NormalizationLayer-3/addAddV2(Encoder-2nd-NormalizationLayer-3/mul:z:0;Encoder-2nd-NormalizationLayer-3/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������P	�
5Encoder-FeedForwardLayer_1_3/Tensordot/ReadVariableOpReadVariableOp>encoder_feedforwardlayer_1_3_tensordot_readvariableop_resource*
_output_shapes

:		*
dtype0u
+Encoder-FeedForwardLayer_1_3/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:|
+Encoder-FeedForwardLayer_1_3/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       �
,Encoder-FeedForwardLayer_1_3/Tensordot/ShapeShape(Encoder-2nd-NormalizationLayer-3/add:z:0*
T0*
_output_shapes
::��v
4Encoder-FeedForwardLayer_1_3/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
/Encoder-FeedForwardLayer_1_3/Tensordot/GatherV2GatherV25Encoder-FeedForwardLayer_1_3/Tensordot/Shape:output:04Encoder-FeedForwardLayer_1_3/Tensordot/free:output:0=Encoder-FeedForwardLayer_1_3/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:x
6Encoder-FeedForwardLayer_1_3/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
1Encoder-FeedForwardLayer_1_3/Tensordot/GatherV2_1GatherV25Encoder-FeedForwardLayer_1_3/Tensordot/Shape:output:04Encoder-FeedForwardLayer_1_3/Tensordot/axes:output:0?Encoder-FeedForwardLayer_1_3/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:v
,Encoder-FeedForwardLayer_1_3/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
+Encoder-FeedForwardLayer_1_3/Tensordot/ProdProd8Encoder-FeedForwardLayer_1_3/Tensordot/GatherV2:output:05Encoder-FeedForwardLayer_1_3/Tensordot/Const:output:0*
T0*
_output_shapes
: x
.Encoder-FeedForwardLayer_1_3/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
-Encoder-FeedForwardLayer_1_3/Tensordot/Prod_1Prod:Encoder-FeedForwardLayer_1_3/Tensordot/GatherV2_1:output:07Encoder-FeedForwardLayer_1_3/Tensordot/Const_1:output:0*
T0*
_output_shapes
: t
2Encoder-FeedForwardLayer_1_3/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
-Encoder-FeedForwardLayer_1_3/Tensordot/concatConcatV24Encoder-FeedForwardLayer_1_3/Tensordot/free:output:04Encoder-FeedForwardLayer_1_3/Tensordot/axes:output:0;Encoder-FeedForwardLayer_1_3/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
,Encoder-FeedForwardLayer_1_3/Tensordot/stackPack4Encoder-FeedForwardLayer_1_3/Tensordot/Prod:output:06Encoder-FeedForwardLayer_1_3/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
0Encoder-FeedForwardLayer_1_3/Tensordot/transpose	Transpose(Encoder-2nd-NormalizationLayer-3/add:z:06Encoder-FeedForwardLayer_1_3/Tensordot/concat:output:0*
T0*+
_output_shapes
:���������P	�
.Encoder-FeedForwardLayer_1_3/Tensordot/ReshapeReshape4Encoder-FeedForwardLayer_1_3/Tensordot/transpose:y:05Encoder-FeedForwardLayer_1_3/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
-Encoder-FeedForwardLayer_1_3/Tensordot/MatMulMatMul7Encoder-FeedForwardLayer_1_3/Tensordot/Reshape:output:0=Encoder-FeedForwardLayer_1_3/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������	x
.Encoder-FeedForwardLayer_1_3/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:	v
4Encoder-FeedForwardLayer_1_3/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
/Encoder-FeedForwardLayer_1_3/Tensordot/concat_1ConcatV28Encoder-FeedForwardLayer_1_3/Tensordot/GatherV2:output:07Encoder-FeedForwardLayer_1_3/Tensordot/Const_2:output:0=Encoder-FeedForwardLayer_1_3/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
&Encoder-FeedForwardLayer_1_3/TensordotReshape7Encoder-FeedForwardLayer_1_3/Tensordot/MatMul:product:08Encoder-FeedForwardLayer_1_3/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������P	�
3Encoder-FeedForwardLayer_1_3/BiasAdd/ReadVariableOpReadVariableOp<encoder_feedforwardlayer_1_3_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype0�
$Encoder-FeedForwardLayer_1_3/BiasAddBiasAdd/Encoder-FeedForwardLayer_1_3/Tensordot:output:0;Encoder-FeedForwardLayer_1_3/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������P	l
'Encoder-FeedForwardLayer_1_3/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
%Encoder-FeedForwardLayer_1_3/Gelu/mulMul0Encoder-FeedForwardLayer_1_3/Gelu/mul/x:output:0-Encoder-FeedForwardLayer_1_3/BiasAdd:output:0*
T0*+
_output_shapes
:���������P	m
(Encoder-FeedForwardLayer_1_3/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *��?�
)Encoder-FeedForwardLayer_1_3/Gelu/truedivRealDiv-Encoder-FeedForwardLayer_1_3/BiasAdd:output:01Encoder-FeedForwardLayer_1_3/Gelu/Cast/x:output:0*
T0*+
_output_shapes
:���������P	�
%Encoder-FeedForwardLayer_1_3/Gelu/ErfErf-Encoder-FeedForwardLayer_1_3/Gelu/truediv:z:0*
T0*+
_output_shapes
:���������P	l
'Encoder-FeedForwardLayer_1_3/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
%Encoder-FeedForwardLayer_1_3/Gelu/addAddV20Encoder-FeedForwardLayer_1_3/Gelu/add/x:output:0)Encoder-FeedForwardLayer_1_3/Gelu/Erf:y:0*
T0*+
_output_shapes
:���������P	�
'Encoder-FeedForwardLayer_1_3/Gelu/mul_1Mul)Encoder-FeedForwardLayer_1_3/Gelu/mul:z:0)Encoder-FeedForwardLayer_1_3/Gelu/add:z:0*
T0*+
_output_shapes
:���������P	�
5Encoder-FeedForwardLayer_2_3/Tensordot/ReadVariableOpReadVariableOp>encoder_feedforwardlayer_2_3_tensordot_readvariableop_resource*
_output_shapes

:		*
dtype0u
+Encoder-FeedForwardLayer_2_3/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:|
+Encoder-FeedForwardLayer_2_3/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       �
,Encoder-FeedForwardLayer_2_3/Tensordot/ShapeShape+Encoder-FeedForwardLayer_1_3/Gelu/mul_1:z:0*
T0*
_output_shapes
::��v
4Encoder-FeedForwardLayer_2_3/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
/Encoder-FeedForwardLayer_2_3/Tensordot/GatherV2GatherV25Encoder-FeedForwardLayer_2_3/Tensordot/Shape:output:04Encoder-FeedForwardLayer_2_3/Tensordot/free:output:0=Encoder-FeedForwardLayer_2_3/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:x
6Encoder-FeedForwardLayer_2_3/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
1Encoder-FeedForwardLayer_2_3/Tensordot/GatherV2_1GatherV25Encoder-FeedForwardLayer_2_3/Tensordot/Shape:output:04Encoder-FeedForwardLayer_2_3/Tensordot/axes:output:0?Encoder-FeedForwardLayer_2_3/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:v
,Encoder-FeedForwardLayer_2_3/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
+Encoder-FeedForwardLayer_2_3/Tensordot/ProdProd8Encoder-FeedForwardLayer_2_3/Tensordot/GatherV2:output:05Encoder-FeedForwardLayer_2_3/Tensordot/Const:output:0*
T0*
_output_shapes
: x
.Encoder-FeedForwardLayer_2_3/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
-Encoder-FeedForwardLayer_2_3/Tensordot/Prod_1Prod:Encoder-FeedForwardLayer_2_3/Tensordot/GatherV2_1:output:07Encoder-FeedForwardLayer_2_3/Tensordot/Const_1:output:0*
T0*
_output_shapes
: t
2Encoder-FeedForwardLayer_2_3/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
-Encoder-FeedForwardLayer_2_3/Tensordot/concatConcatV24Encoder-FeedForwardLayer_2_3/Tensordot/free:output:04Encoder-FeedForwardLayer_2_3/Tensordot/axes:output:0;Encoder-FeedForwardLayer_2_3/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
,Encoder-FeedForwardLayer_2_3/Tensordot/stackPack4Encoder-FeedForwardLayer_2_3/Tensordot/Prod:output:06Encoder-FeedForwardLayer_2_3/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
0Encoder-FeedForwardLayer_2_3/Tensordot/transpose	Transpose+Encoder-FeedForwardLayer_1_3/Gelu/mul_1:z:06Encoder-FeedForwardLayer_2_3/Tensordot/concat:output:0*
T0*+
_output_shapes
:���������P	�
.Encoder-FeedForwardLayer_2_3/Tensordot/ReshapeReshape4Encoder-FeedForwardLayer_2_3/Tensordot/transpose:y:05Encoder-FeedForwardLayer_2_3/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
-Encoder-FeedForwardLayer_2_3/Tensordot/MatMulMatMul7Encoder-FeedForwardLayer_2_3/Tensordot/Reshape:output:0=Encoder-FeedForwardLayer_2_3/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������	x
.Encoder-FeedForwardLayer_2_3/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:	v
4Encoder-FeedForwardLayer_2_3/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
/Encoder-FeedForwardLayer_2_3/Tensordot/concat_1ConcatV28Encoder-FeedForwardLayer_2_3/Tensordot/GatherV2:output:07Encoder-FeedForwardLayer_2_3/Tensordot/Const_2:output:0=Encoder-FeedForwardLayer_2_3/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
&Encoder-FeedForwardLayer_2_3/TensordotReshape7Encoder-FeedForwardLayer_2_3/Tensordot/MatMul:product:08Encoder-FeedForwardLayer_2_3/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������P	�
3Encoder-FeedForwardLayer_2_3/BiasAdd/ReadVariableOpReadVariableOp<encoder_feedforwardlayer_2_3_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype0�
$Encoder-FeedForwardLayer_2_3/BiasAddBiasAdd/Encoder-FeedForwardLayer_2_3/Tensordot:output:0;Encoder-FeedForwardLayer_2_3/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������P	�
dropout_5/IdentityIdentity-Encoder-FeedForwardLayer_2_3/BiasAdd:output:0*
T0*+
_output_shapes
:���������P	�
Encoder-2nd-AdditionLayer-3/addAddV2#Encoder-1st-AdditionLayer-3/add:z:0dropout_5/Identity:output:0*
T0*+
_output_shapes
:���������P	v
IdentityIdentity#Encoder-2nd-AdditionLayer-3/add:z:0^NoOp*
T0*+
_output_shapes
:���������P	�

Identity_1Identity6Encoder-SelfAttentionLayer-3/softmax/Softmax:softmax:0^NoOp*
T0*/
_output_shapes
:���������PP�
NoOpNoOp4^Encoder-1st-NormalizationLayer-3/add/ReadVariableOp4^Encoder-1st-NormalizationLayer-3/mul/ReadVariableOp4^Encoder-2nd-NormalizationLayer-3/add/ReadVariableOp4^Encoder-2nd-NormalizationLayer-3/mul/ReadVariableOp4^Encoder-FeedForwardLayer_1_3/BiasAdd/ReadVariableOp6^Encoder-FeedForwardLayer_1_3/Tensordot/ReadVariableOp4^Encoder-FeedForwardLayer_2_3/BiasAdd/ReadVariableOp6^Encoder-FeedForwardLayer_2_3/Tensordot/ReadVariableOpA^Encoder-SelfAttentionLayer-3/attention_output/add/ReadVariableOpK^Encoder-SelfAttentionLayer-3/attention_output/einsum/Einsum/ReadVariableOp4^Encoder-SelfAttentionLayer-3/key/add/ReadVariableOp>^Encoder-SelfAttentionLayer-3/key/einsum/Einsum/ReadVariableOp6^Encoder-SelfAttentionLayer-3/query/add/ReadVariableOp@^Encoder-SelfAttentionLayer-3/query/einsum/Einsum/ReadVariableOp6^Encoder-SelfAttentionLayer-3/value/add/ReadVariableOp@^Encoder-SelfAttentionLayer-3/value/einsum/Einsum/ReadVariableOp*
_output_shapes
 "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:���������P	: : : : : : : : : : : : : : : : 2j
3Encoder-1st-NormalizationLayer-3/add/ReadVariableOp3Encoder-1st-NormalizationLayer-3/add/ReadVariableOp2j
3Encoder-1st-NormalizationLayer-3/mul/ReadVariableOp3Encoder-1st-NormalizationLayer-3/mul/ReadVariableOp2j
3Encoder-2nd-NormalizationLayer-3/add/ReadVariableOp3Encoder-2nd-NormalizationLayer-3/add/ReadVariableOp2j
3Encoder-2nd-NormalizationLayer-3/mul/ReadVariableOp3Encoder-2nd-NormalizationLayer-3/mul/ReadVariableOp2j
3Encoder-FeedForwardLayer_1_3/BiasAdd/ReadVariableOp3Encoder-FeedForwardLayer_1_3/BiasAdd/ReadVariableOp2n
5Encoder-FeedForwardLayer_1_3/Tensordot/ReadVariableOp5Encoder-FeedForwardLayer_1_3/Tensordot/ReadVariableOp2j
3Encoder-FeedForwardLayer_2_3/BiasAdd/ReadVariableOp3Encoder-FeedForwardLayer_2_3/BiasAdd/ReadVariableOp2n
5Encoder-FeedForwardLayer_2_3/Tensordot/ReadVariableOp5Encoder-FeedForwardLayer_2_3/Tensordot/ReadVariableOp2�
@Encoder-SelfAttentionLayer-3/attention_output/add/ReadVariableOp@Encoder-SelfAttentionLayer-3/attention_output/add/ReadVariableOp2�
JEncoder-SelfAttentionLayer-3/attention_output/einsum/Einsum/ReadVariableOpJEncoder-SelfAttentionLayer-3/attention_output/einsum/Einsum/ReadVariableOp2j
3Encoder-SelfAttentionLayer-3/key/add/ReadVariableOp3Encoder-SelfAttentionLayer-3/key/add/ReadVariableOp2~
=Encoder-SelfAttentionLayer-3/key/einsum/Einsum/ReadVariableOp=Encoder-SelfAttentionLayer-3/key/einsum/Einsum/ReadVariableOp2n
5Encoder-SelfAttentionLayer-3/query/add/ReadVariableOp5Encoder-SelfAttentionLayer-3/query/add/ReadVariableOp2�
?Encoder-SelfAttentionLayer-3/query/einsum/Einsum/ReadVariableOp?Encoder-SelfAttentionLayer-3/query/einsum/Einsum/ReadVariableOp2n
5Encoder-SelfAttentionLayer-3/value/add/ReadVariableOp5Encoder-SelfAttentionLayer-3/value/add/ReadVariableOp2�
?Encoder-SelfAttentionLayer-3/value/einsum/Einsum/ReadVariableOp?Encoder-SelfAttentionLayer-3/value/einsum/Einsum/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:(
$
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:S O
+
_output_shapes
:���������P	
 
_user_specified_nameinputs
�
�
Z__inference_FullyConnectedLayerImprovement_layer_call_and_return_conditional_losses_233352

inputs0
matmul_readvariableop_resource:

-
biasadd_readvariableop_resource:

identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:

*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
O

Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?h
Gelu/mulMulGelu/mul/x:output:0BiasAdd:output:0*
T0*'
_output_shapes
:���������
P
Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *��?q
Gelu/truedivRealDivBiasAdd:output:0Gelu/Cast/x:output:0*
T0*'
_output_shapes
:���������
S
Gelu/ErfErfGelu/truediv:z:0*
T0*'
_output_shapes
:���������
O

Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?f
Gelu/addAddV2Gelu/add/x:output:0Gelu/Erf:y:0*
T0*'
_output_shapes
:���������
_

Gelu/mul_1MulGelu/mul:z:0Gelu/add:z:0*
T0*'
_output_shapes
:���������
]
IdentityIdentityGelu/mul_1:z:0^NoOp*
T0*'
_output_shapes
:���������
S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
'
_output_shapes
:���������

 
_user_specified_nameinputs
��
�
Q__inference_transformer_encoder_4_layer_call_and_return_conditional_losses_230943

inputsJ
<encoder_1st_normalizationlayer_2_mul_readvariableop_resource:	J
<encoder_1st_normalizationlayer_2_add_readvariableop_resource:	^
Hencoder_selfattentionlayer_2_query_einsum_einsum_readvariableop_resource:	P
>encoder_selfattentionlayer_2_query_add_readvariableop_resource:\
Fencoder_selfattentionlayer_2_key_einsum_einsum_readvariableop_resource:	N
<encoder_selfattentionlayer_2_key_add_readvariableop_resource:^
Hencoder_selfattentionlayer_2_value_einsum_einsum_readvariableop_resource:	P
>encoder_selfattentionlayer_2_value_add_readvariableop_resource:i
Sencoder_selfattentionlayer_2_attention_output_einsum_einsum_readvariableop_resource:	W
Iencoder_selfattentionlayer_2_attention_output_add_readvariableop_resource:	J
<encoder_2nd_normalizationlayer_2_mul_readvariableop_resource:	J
<encoder_2nd_normalizationlayer_2_add_readvariableop_resource:	P
>encoder_feedforwardlayer_1_2_tensordot_readvariableop_resource:		J
<encoder_feedforwardlayer_1_2_biasadd_readvariableop_resource:	P
>encoder_feedforwardlayer_2_2_tensordot_readvariableop_resource:		J
<encoder_feedforwardlayer_2_2_biasadd_readvariableop_resource:	
identity

identity_1��3Encoder-1st-NormalizationLayer-2/add/ReadVariableOp�3Encoder-1st-NormalizationLayer-2/mul/ReadVariableOp�3Encoder-2nd-NormalizationLayer-2/add/ReadVariableOp�3Encoder-2nd-NormalizationLayer-2/mul/ReadVariableOp�3Encoder-FeedForwardLayer_1_2/BiasAdd/ReadVariableOp�5Encoder-FeedForwardLayer_1_2/Tensordot/ReadVariableOp�3Encoder-FeedForwardLayer_2_2/BiasAdd/ReadVariableOp�5Encoder-FeedForwardLayer_2_2/Tensordot/ReadVariableOp�@Encoder-SelfAttentionLayer-2/attention_output/add/ReadVariableOp�JEncoder-SelfAttentionLayer-2/attention_output/einsum/Einsum/ReadVariableOp�3Encoder-SelfAttentionLayer-2/key/add/ReadVariableOp�=Encoder-SelfAttentionLayer-2/key/einsum/Einsum/ReadVariableOp�5Encoder-SelfAttentionLayer-2/query/add/ReadVariableOp�?Encoder-SelfAttentionLayer-2/query/einsum/Einsum/ReadVariableOp�5Encoder-SelfAttentionLayer-2/value/add/ReadVariableOp�?Encoder-SelfAttentionLayer-2/value/einsum/Einsum/ReadVariableOpj
&Encoder-1st-NormalizationLayer-2/ShapeShapeinputs*
T0*
_output_shapes
::��~
4Encoder-1st-NormalizationLayer-2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
6Encoder-1st-NormalizationLayer-2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
6Encoder-1st-NormalizationLayer-2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
.Encoder-1st-NormalizationLayer-2/strided_sliceStridedSlice/Encoder-1st-NormalizationLayer-2/Shape:output:0=Encoder-1st-NormalizationLayer-2/strided_slice/stack:output:0?Encoder-1st-NormalizationLayer-2/strided_slice/stack_1:output:0?Encoder-1st-NormalizationLayer-2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskp
&Encoder-1st-NormalizationLayer-2/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
%Encoder-1st-NormalizationLayer-2/ProdProd7Encoder-1st-NormalizationLayer-2/strided_slice:output:0/Encoder-1st-NormalizationLayer-2/Const:output:0*
T0*
_output_shapes
: �
6Encoder-1st-NormalizationLayer-2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:�
8Encoder-1st-NormalizationLayer-2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: �
8Encoder-1st-NormalizationLayer-2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
0Encoder-1st-NormalizationLayer-2/strided_slice_1StridedSlice/Encoder-1st-NormalizationLayer-2/Shape:output:0?Encoder-1st-NormalizationLayer-2/strided_slice_1/stack:output:0AEncoder-1st-NormalizationLayer-2/strided_slice_1/stack_1:output:0AEncoder-1st-NormalizationLayer-2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_maskr
(Encoder-1st-NormalizationLayer-2/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
'Encoder-1st-NormalizationLayer-2/Prod_1Prod9Encoder-1st-NormalizationLayer-2/strided_slice_1:output:01Encoder-1st-NormalizationLayer-2/Const_1:output:0*
T0*
_output_shapes
: r
0Encoder-1st-NormalizationLayer-2/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :r
0Encoder-1st-NormalizationLayer-2/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :�
.Encoder-1st-NormalizationLayer-2/Reshape/shapePack9Encoder-1st-NormalizationLayer-2/Reshape/shape/0:output:0.Encoder-1st-NormalizationLayer-2/Prod:output:00Encoder-1st-NormalizationLayer-2/Prod_1:output:09Encoder-1st-NormalizationLayer-2/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:�
(Encoder-1st-NormalizationLayer-2/ReshapeReshapeinputs7Encoder-1st-NormalizationLayer-2/Reshape/shape:output:0*
T0*8
_output_shapes&
$:"�������������������
,Encoder-1st-NormalizationLayer-2/ones/packedPack.Encoder-1st-NormalizationLayer-2/Prod:output:0*
N*
T0*
_output_shapes
:p
+Encoder-1st-NormalizationLayer-2/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
%Encoder-1st-NormalizationLayer-2/onesFill5Encoder-1st-NormalizationLayer-2/ones/packed:output:04Encoder-1st-NormalizationLayer-2/ones/Const:output:0*
T0*#
_output_shapes
:����������
-Encoder-1st-NormalizationLayer-2/zeros/packedPack.Encoder-1st-NormalizationLayer-2/Prod:output:0*
N*
T0*
_output_shapes
:q
,Encoder-1st-NormalizationLayer-2/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
&Encoder-1st-NormalizationLayer-2/zerosFill6Encoder-1st-NormalizationLayer-2/zeros/packed:output:05Encoder-1st-NormalizationLayer-2/zeros/Const:output:0*
T0*#
_output_shapes
:���������k
(Encoder-1st-NormalizationLayer-2/Const_2Const*
_output_shapes
: *
dtype0*
valueB k
(Encoder-1st-NormalizationLayer-2/Const_3Const*
_output_shapes
: *
dtype0*
valueB �
1Encoder-1st-NormalizationLayer-2/FusedBatchNormV3FusedBatchNormV31Encoder-1st-NormalizationLayer-2/Reshape:output:0.Encoder-1st-NormalizationLayer-2/ones:output:0/Encoder-1st-NormalizationLayer-2/zeros:output:01Encoder-1st-NormalizationLayer-2/Const_2:output:01Encoder-1st-NormalizationLayer-2/Const_3:output:0*
T0*
U0*x
_output_shapesf
d:"������������������:���������:���������:���������:���������:*
data_formatNCHW*
epsilon%o�:�
*Encoder-1st-NormalizationLayer-2/Reshape_1Reshape5Encoder-1st-NormalizationLayer-2/FusedBatchNormV3:y:0/Encoder-1st-NormalizationLayer-2/Shape:output:0*
T0*+
_output_shapes
:���������P	�
3Encoder-1st-NormalizationLayer-2/mul/ReadVariableOpReadVariableOp<encoder_1st_normalizationlayer_2_mul_readvariableop_resource*
_output_shapes
:	*
dtype0�
$Encoder-1st-NormalizationLayer-2/mulMul3Encoder-1st-NormalizationLayer-2/Reshape_1:output:0;Encoder-1st-NormalizationLayer-2/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������P	�
3Encoder-1st-NormalizationLayer-2/add/ReadVariableOpReadVariableOp<encoder_1st_normalizationlayer_2_add_readvariableop_resource*
_output_shapes
:	*
dtype0�
$Encoder-1st-NormalizationLayer-2/addAddV2(Encoder-1st-NormalizationLayer-2/mul:z:0;Encoder-1st-NormalizationLayer-2/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������P	�
?Encoder-SelfAttentionLayer-2/query/einsum/Einsum/ReadVariableOpReadVariableOpHencoder_selfattentionlayer_2_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:	*
dtype0�
0Encoder-SelfAttentionLayer-2/query/einsum/EinsumEinsum(Encoder-1st-NormalizationLayer-2/add:z:0GEncoder-SelfAttentionLayer-2/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������P*
equationabc,cde->abde�
5Encoder-SelfAttentionLayer-2/query/add/ReadVariableOpReadVariableOp>encoder_selfattentionlayer_2_query_add_readvariableop_resource*
_output_shapes

:*
dtype0�
&Encoder-SelfAttentionLayer-2/query/addAddV29Encoder-SelfAttentionLayer-2/query/einsum/Einsum:output:0=Encoder-SelfAttentionLayer-2/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������P�
=Encoder-SelfAttentionLayer-2/key/einsum/Einsum/ReadVariableOpReadVariableOpFencoder_selfattentionlayer_2_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:	*
dtype0�
.Encoder-SelfAttentionLayer-2/key/einsum/EinsumEinsum(Encoder-1st-NormalizationLayer-2/add:z:0EEncoder-SelfAttentionLayer-2/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������P*
equationabc,cde->abde�
3Encoder-SelfAttentionLayer-2/key/add/ReadVariableOpReadVariableOp<encoder_selfattentionlayer_2_key_add_readvariableop_resource*
_output_shapes

:*
dtype0�
$Encoder-SelfAttentionLayer-2/key/addAddV27Encoder-SelfAttentionLayer-2/key/einsum/Einsum:output:0;Encoder-SelfAttentionLayer-2/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������P�
?Encoder-SelfAttentionLayer-2/value/einsum/Einsum/ReadVariableOpReadVariableOpHencoder_selfattentionlayer_2_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:	*
dtype0�
0Encoder-SelfAttentionLayer-2/value/einsum/EinsumEinsum(Encoder-1st-NormalizationLayer-2/add:z:0GEncoder-SelfAttentionLayer-2/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������P*
equationabc,cde->abde�
5Encoder-SelfAttentionLayer-2/value/add/ReadVariableOpReadVariableOp>encoder_selfattentionlayer_2_value_add_readvariableop_resource*
_output_shapes

:*
dtype0�
&Encoder-SelfAttentionLayer-2/value/addAddV29Encoder-SelfAttentionLayer-2/value/einsum/Einsum:output:0=Encoder-SelfAttentionLayer-2/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������Pg
"Encoder-SelfAttentionLayer-2/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *:�?�
 Encoder-SelfAttentionLayer-2/MulMul*Encoder-SelfAttentionLayer-2/query/add:z:0+Encoder-SelfAttentionLayer-2/Mul/y:output:0*
T0*/
_output_shapes
:���������P�
*Encoder-SelfAttentionLayer-2/einsum/EinsumEinsum(Encoder-SelfAttentionLayer-2/key/add:z:0$Encoder-SelfAttentionLayer-2/Mul:z:0*
N*
T0*/
_output_shapes
:���������PP*
equationaecd,abcd->acbe�
,Encoder-SelfAttentionLayer-2/softmax/SoftmaxSoftmax3Encoder-SelfAttentionLayer-2/einsum/Einsum:output:0*
T0*/
_output_shapes
:���������PP�
-Encoder-SelfAttentionLayer-2/dropout/IdentityIdentity6Encoder-SelfAttentionLayer-2/softmax/Softmax:softmax:0*
T0*/
_output_shapes
:���������PP�
,Encoder-SelfAttentionLayer-2/einsum_1/EinsumEinsum6Encoder-SelfAttentionLayer-2/dropout/Identity:output:0*Encoder-SelfAttentionLayer-2/value/add:z:0*
N*
T0*/
_output_shapes
:���������P*
equationacbe,aecd->abcd�
JEncoder-SelfAttentionLayer-2/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpSencoder_selfattentionlayer_2_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:	*
dtype0�
;Encoder-SelfAttentionLayer-2/attention_output/einsum/EinsumEinsum5Encoder-SelfAttentionLayer-2/einsum_1/Einsum:output:0REncoder-SelfAttentionLayer-2/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:���������P	*
equationabcd,cde->abe�
@Encoder-SelfAttentionLayer-2/attention_output/add/ReadVariableOpReadVariableOpIencoder_selfattentionlayer_2_attention_output_add_readvariableop_resource*
_output_shapes
:	*
dtype0�
1Encoder-SelfAttentionLayer-2/attention_output/addAddV2DEncoder-SelfAttentionLayer-2/attention_output/einsum/Einsum:output:0HEncoder-SelfAttentionLayer-2/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������P	�
Encoder-1st-AdditionLayer-2/addAddV2inputs5Encoder-SelfAttentionLayer-2/attention_output/add:z:0*
T0*+
_output_shapes
:���������P	�
&Encoder-2nd-NormalizationLayer-2/ShapeShape#Encoder-1st-AdditionLayer-2/add:z:0*
T0*
_output_shapes
::��~
4Encoder-2nd-NormalizationLayer-2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
6Encoder-2nd-NormalizationLayer-2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
6Encoder-2nd-NormalizationLayer-2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
.Encoder-2nd-NormalizationLayer-2/strided_sliceStridedSlice/Encoder-2nd-NormalizationLayer-2/Shape:output:0=Encoder-2nd-NormalizationLayer-2/strided_slice/stack:output:0?Encoder-2nd-NormalizationLayer-2/strided_slice/stack_1:output:0?Encoder-2nd-NormalizationLayer-2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskp
&Encoder-2nd-NormalizationLayer-2/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
%Encoder-2nd-NormalizationLayer-2/ProdProd7Encoder-2nd-NormalizationLayer-2/strided_slice:output:0/Encoder-2nd-NormalizationLayer-2/Const:output:0*
T0*
_output_shapes
: �
6Encoder-2nd-NormalizationLayer-2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:�
8Encoder-2nd-NormalizationLayer-2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: �
8Encoder-2nd-NormalizationLayer-2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
0Encoder-2nd-NormalizationLayer-2/strided_slice_1StridedSlice/Encoder-2nd-NormalizationLayer-2/Shape:output:0?Encoder-2nd-NormalizationLayer-2/strided_slice_1/stack:output:0AEncoder-2nd-NormalizationLayer-2/strided_slice_1/stack_1:output:0AEncoder-2nd-NormalizationLayer-2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_maskr
(Encoder-2nd-NormalizationLayer-2/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
'Encoder-2nd-NormalizationLayer-2/Prod_1Prod9Encoder-2nd-NormalizationLayer-2/strided_slice_1:output:01Encoder-2nd-NormalizationLayer-2/Const_1:output:0*
T0*
_output_shapes
: r
0Encoder-2nd-NormalizationLayer-2/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :r
0Encoder-2nd-NormalizationLayer-2/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :�
.Encoder-2nd-NormalizationLayer-2/Reshape/shapePack9Encoder-2nd-NormalizationLayer-2/Reshape/shape/0:output:0.Encoder-2nd-NormalizationLayer-2/Prod:output:00Encoder-2nd-NormalizationLayer-2/Prod_1:output:09Encoder-2nd-NormalizationLayer-2/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:�
(Encoder-2nd-NormalizationLayer-2/ReshapeReshape#Encoder-1st-AdditionLayer-2/add:z:07Encoder-2nd-NormalizationLayer-2/Reshape/shape:output:0*
T0*8
_output_shapes&
$:"�������������������
,Encoder-2nd-NormalizationLayer-2/ones/packedPack.Encoder-2nd-NormalizationLayer-2/Prod:output:0*
N*
T0*
_output_shapes
:p
+Encoder-2nd-NormalizationLayer-2/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
%Encoder-2nd-NormalizationLayer-2/onesFill5Encoder-2nd-NormalizationLayer-2/ones/packed:output:04Encoder-2nd-NormalizationLayer-2/ones/Const:output:0*
T0*#
_output_shapes
:����������
-Encoder-2nd-NormalizationLayer-2/zeros/packedPack.Encoder-2nd-NormalizationLayer-2/Prod:output:0*
N*
T0*
_output_shapes
:q
,Encoder-2nd-NormalizationLayer-2/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
&Encoder-2nd-NormalizationLayer-2/zerosFill6Encoder-2nd-NormalizationLayer-2/zeros/packed:output:05Encoder-2nd-NormalizationLayer-2/zeros/Const:output:0*
T0*#
_output_shapes
:���������k
(Encoder-2nd-NormalizationLayer-2/Const_2Const*
_output_shapes
: *
dtype0*
valueB k
(Encoder-2nd-NormalizationLayer-2/Const_3Const*
_output_shapes
: *
dtype0*
valueB �
1Encoder-2nd-NormalizationLayer-2/FusedBatchNormV3FusedBatchNormV31Encoder-2nd-NormalizationLayer-2/Reshape:output:0.Encoder-2nd-NormalizationLayer-2/ones:output:0/Encoder-2nd-NormalizationLayer-2/zeros:output:01Encoder-2nd-NormalizationLayer-2/Const_2:output:01Encoder-2nd-NormalizationLayer-2/Const_3:output:0*
T0*
U0*x
_output_shapesf
d:"������������������:���������:���������:���������:���������:*
data_formatNCHW*
epsilon%o�:�
*Encoder-2nd-NormalizationLayer-2/Reshape_1Reshape5Encoder-2nd-NormalizationLayer-2/FusedBatchNormV3:y:0/Encoder-2nd-NormalizationLayer-2/Shape:output:0*
T0*+
_output_shapes
:���������P	�
3Encoder-2nd-NormalizationLayer-2/mul/ReadVariableOpReadVariableOp<encoder_2nd_normalizationlayer_2_mul_readvariableop_resource*
_output_shapes
:	*
dtype0�
$Encoder-2nd-NormalizationLayer-2/mulMul3Encoder-2nd-NormalizationLayer-2/Reshape_1:output:0;Encoder-2nd-NormalizationLayer-2/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������P	�
3Encoder-2nd-NormalizationLayer-2/add/ReadVariableOpReadVariableOp<encoder_2nd_normalizationlayer_2_add_readvariableop_resource*
_output_shapes
:	*
dtype0�
$Encoder-2nd-NormalizationLayer-2/addAddV2(Encoder-2nd-NormalizationLayer-2/mul:z:0;Encoder-2nd-NormalizationLayer-2/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������P	�
5Encoder-FeedForwardLayer_1_2/Tensordot/ReadVariableOpReadVariableOp>encoder_feedforwardlayer_1_2_tensordot_readvariableop_resource*
_output_shapes

:		*
dtype0u
+Encoder-FeedForwardLayer_1_2/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:|
+Encoder-FeedForwardLayer_1_2/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       �
,Encoder-FeedForwardLayer_1_2/Tensordot/ShapeShape(Encoder-2nd-NormalizationLayer-2/add:z:0*
T0*
_output_shapes
::��v
4Encoder-FeedForwardLayer_1_2/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
/Encoder-FeedForwardLayer_1_2/Tensordot/GatherV2GatherV25Encoder-FeedForwardLayer_1_2/Tensordot/Shape:output:04Encoder-FeedForwardLayer_1_2/Tensordot/free:output:0=Encoder-FeedForwardLayer_1_2/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:x
6Encoder-FeedForwardLayer_1_2/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
1Encoder-FeedForwardLayer_1_2/Tensordot/GatherV2_1GatherV25Encoder-FeedForwardLayer_1_2/Tensordot/Shape:output:04Encoder-FeedForwardLayer_1_2/Tensordot/axes:output:0?Encoder-FeedForwardLayer_1_2/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:v
,Encoder-FeedForwardLayer_1_2/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
+Encoder-FeedForwardLayer_1_2/Tensordot/ProdProd8Encoder-FeedForwardLayer_1_2/Tensordot/GatherV2:output:05Encoder-FeedForwardLayer_1_2/Tensordot/Const:output:0*
T0*
_output_shapes
: x
.Encoder-FeedForwardLayer_1_2/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
-Encoder-FeedForwardLayer_1_2/Tensordot/Prod_1Prod:Encoder-FeedForwardLayer_1_2/Tensordot/GatherV2_1:output:07Encoder-FeedForwardLayer_1_2/Tensordot/Const_1:output:0*
T0*
_output_shapes
: t
2Encoder-FeedForwardLayer_1_2/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
-Encoder-FeedForwardLayer_1_2/Tensordot/concatConcatV24Encoder-FeedForwardLayer_1_2/Tensordot/free:output:04Encoder-FeedForwardLayer_1_2/Tensordot/axes:output:0;Encoder-FeedForwardLayer_1_2/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
,Encoder-FeedForwardLayer_1_2/Tensordot/stackPack4Encoder-FeedForwardLayer_1_2/Tensordot/Prod:output:06Encoder-FeedForwardLayer_1_2/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
0Encoder-FeedForwardLayer_1_2/Tensordot/transpose	Transpose(Encoder-2nd-NormalizationLayer-2/add:z:06Encoder-FeedForwardLayer_1_2/Tensordot/concat:output:0*
T0*+
_output_shapes
:���������P	�
.Encoder-FeedForwardLayer_1_2/Tensordot/ReshapeReshape4Encoder-FeedForwardLayer_1_2/Tensordot/transpose:y:05Encoder-FeedForwardLayer_1_2/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
-Encoder-FeedForwardLayer_1_2/Tensordot/MatMulMatMul7Encoder-FeedForwardLayer_1_2/Tensordot/Reshape:output:0=Encoder-FeedForwardLayer_1_2/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������	x
.Encoder-FeedForwardLayer_1_2/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:	v
4Encoder-FeedForwardLayer_1_2/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
/Encoder-FeedForwardLayer_1_2/Tensordot/concat_1ConcatV28Encoder-FeedForwardLayer_1_2/Tensordot/GatherV2:output:07Encoder-FeedForwardLayer_1_2/Tensordot/Const_2:output:0=Encoder-FeedForwardLayer_1_2/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
&Encoder-FeedForwardLayer_1_2/TensordotReshape7Encoder-FeedForwardLayer_1_2/Tensordot/MatMul:product:08Encoder-FeedForwardLayer_1_2/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������P	�
3Encoder-FeedForwardLayer_1_2/BiasAdd/ReadVariableOpReadVariableOp<encoder_feedforwardlayer_1_2_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype0�
$Encoder-FeedForwardLayer_1_2/BiasAddBiasAdd/Encoder-FeedForwardLayer_1_2/Tensordot:output:0;Encoder-FeedForwardLayer_1_2/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������P	l
'Encoder-FeedForwardLayer_1_2/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
%Encoder-FeedForwardLayer_1_2/Gelu/mulMul0Encoder-FeedForwardLayer_1_2/Gelu/mul/x:output:0-Encoder-FeedForwardLayer_1_2/BiasAdd:output:0*
T0*+
_output_shapes
:���������P	m
(Encoder-FeedForwardLayer_1_2/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *��?�
)Encoder-FeedForwardLayer_1_2/Gelu/truedivRealDiv-Encoder-FeedForwardLayer_1_2/BiasAdd:output:01Encoder-FeedForwardLayer_1_2/Gelu/Cast/x:output:0*
T0*+
_output_shapes
:���������P	�
%Encoder-FeedForwardLayer_1_2/Gelu/ErfErf-Encoder-FeedForwardLayer_1_2/Gelu/truediv:z:0*
T0*+
_output_shapes
:���������P	l
'Encoder-FeedForwardLayer_1_2/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
%Encoder-FeedForwardLayer_1_2/Gelu/addAddV20Encoder-FeedForwardLayer_1_2/Gelu/add/x:output:0)Encoder-FeedForwardLayer_1_2/Gelu/Erf:y:0*
T0*+
_output_shapes
:���������P	�
'Encoder-FeedForwardLayer_1_2/Gelu/mul_1Mul)Encoder-FeedForwardLayer_1_2/Gelu/mul:z:0)Encoder-FeedForwardLayer_1_2/Gelu/add:z:0*
T0*+
_output_shapes
:���������P	�
5Encoder-FeedForwardLayer_2_2/Tensordot/ReadVariableOpReadVariableOp>encoder_feedforwardlayer_2_2_tensordot_readvariableop_resource*
_output_shapes

:		*
dtype0u
+Encoder-FeedForwardLayer_2_2/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:|
+Encoder-FeedForwardLayer_2_2/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       �
,Encoder-FeedForwardLayer_2_2/Tensordot/ShapeShape+Encoder-FeedForwardLayer_1_2/Gelu/mul_1:z:0*
T0*
_output_shapes
::��v
4Encoder-FeedForwardLayer_2_2/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
/Encoder-FeedForwardLayer_2_2/Tensordot/GatherV2GatherV25Encoder-FeedForwardLayer_2_2/Tensordot/Shape:output:04Encoder-FeedForwardLayer_2_2/Tensordot/free:output:0=Encoder-FeedForwardLayer_2_2/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:x
6Encoder-FeedForwardLayer_2_2/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
1Encoder-FeedForwardLayer_2_2/Tensordot/GatherV2_1GatherV25Encoder-FeedForwardLayer_2_2/Tensordot/Shape:output:04Encoder-FeedForwardLayer_2_2/Tensordot/axes:output:0?Encoder-FeedForwardLayer_2_2/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:v
,Encoder-FeedForwardLayer_2_2/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
+Encoder-FeedForwardLayer_2_2/Tensordot/ProdProd8Encoder-FeedForwardLayer_2_2/Tensordot/GatherV2:output:05Encoder-FeedForwardLayer_2_2/Tensordot/Const:output:0*
T0*
_output_shapes
: x
.Encoder-FeedForwardLayer_2_2/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
-Encoder-FeedForwardLayer_2_2/Tensordot/Prod_1Prod:Encoder-FeedForwardLayer_2_2/Tensordot/GatherV2_1:output:07Encoder-FeedForwardLayer_2_2/Tensordot/Const_1:output:0*
T0*
_output_shapes
: t
2Encoder-FeedForwardLayer_2_2/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
-Encoder-FeedForwardLayer_2_2/Tensordot/concatConcatV24Encoder-FeedForwardLayer_2_2/Tensordot/free:output:04Encoder-FeedForwardLayer_2_2/Tensordot/axes:output:0;Encoder-FeedForwardLayer_2_2/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
,Encoder-FeedForwardLayer_2_2/Tensordot/stackPack4Encoder-FeedForwardLayer_2_2/Tensordot/Prod:output:06Encoder-FeedForwardLayer_2_2/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
0Encoder-FeedForwardLayer_2_2/Tensordot/transpose	Transpose+Encoder-FeedForwardLayer_1_2/Gelu/mul_1:z:06Encoder-FeedForwardLayer_2_2/Tensordot/concat:output:0*
T0*+
_output_shapes
:���������P	�
.Encoder-FeedForwardLayer_2_2/Tensordot/ReshapeReshape4Encoder-FeedForwardLayer_2_2/Tensordot/transpose:y:05Encoder-FeedForwardLayer_2_2/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
-Encoder-FeedForwardLayer_2_2/Tensordot/MatMulMatMul7Encoder-FeedForwardLayer_2_2/Tensordot/Reshape:output:0=Encoder-FeedForwardLayer_2_2/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������	x
.Encoder-FeedForwardLayer_2_2/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:	v
4Encoder-FeedForwardLayer_2_2/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
/Encoder-FeedForwardLayer_2_2/Tensordot/concat_1ConcatV28Encoder-FeedForwardLayer_2_2/Tensordot/GatherV2:output:07Encoder-FeedForwardLayer_2_2/Tensordot/Const_2:output:0=Encoder-FeedForwardLayer_2_2/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
&Encoder-FeedForwardLayer_2_2/TensordotReshape7Encoder-FeedForwardLayer_2_2/Tensordot/MatMul:product:08Encoder-FeedForwardLayer_2_2/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������P	�
3Encoder-FeedForwardLayer_2_2/BiasAdd/ReadVariableOpReadVariableOp<encoder_feedforwardlayer_2_2_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype0�
$Encoder-FeedForwardLayer_2_2/BiasAddBiasAdd/Encoder-FeedForwardLayer_2_2/Tensordot:output:0;Encoder-FeedForwardLayer_2_2/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������P	�
dropout_4/IdentityIdentity-Encoder-FeedForwardLayer_2_2/BiasAdd:output:0*
T0*+
_output_shapes
:���������P	�
Encoder-2nd-AdditionLayer-2/addAddV2#Encoder-1st-AdditionLayer-2/add:z:0dropout_4/Identity:output:0*
T0*+
_output_shapes
:���������P	v
IdentityIdentity#Encoder-2nd-AdditionLayer-2/add:z:0^NoOp*
T0*+
_output_shapes
:���������P	�

Identity_1Identity6Encoder-SelfAttentionLayer-2/softmax/Softmax:softmax:0^NoOp*
T0*/
_output_shapes
:���������PP�
NoOpNoOp4^Encoder-1st-NormalizationLayer-2/add/ReadVariableOp4^Encoder-1st-NormalizationLayer-2/mul/ReadVariableOp4^Encoder-2nd-NormalizationLayer-2/add/ReadVariableOp4^Encoder-2nd-NormalizationLayer-2/mul/ReadVariableOp4^Encoder-FeedForwardLayer_1_2/BiasAdd/ReadVariableOp6^Encoder-FeedForwardLayer_1_2/Tensordot/ReadVariableOp4^Encoder-FeedForwardLayer_2_2/BiasAdd/ReadVariableOp6^Encoder-FeedForwardLayer_2_2/Tensordot/ReadVariableOpA^Encoder-SelfAttentionLayer-2/attention_output/add/ReadVariableOpK^Encoder-SelfAttentionLayer-2/attention_output/einsum/Einsum/ReadVariableOp4^Encoder-SelfAttentionLayer-2/key/add/ReadVariableOp>^Encoder-SelfAttentionLayer-2/key/einsum/Einsum/ReadVariableOp6^Encoder-SelfAttentionLayer-2/query/add/ReadVariableOp@^Encoder-SelfAttentionLayer-2/query/einsum/Einsum/ReadVariableOp6^Encoder-SelfAttentionLayer-2/value/add/ReadVariableOp@^Encoder-SelfAttentionLayer-2/value/einsum/Einsum/ReadVariableOp*
_output_shapes
 "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:���������P	: : : : : : : : : : : : : : : : 2j
3Encoder-1st-NormalizationLayer-2/add/ReadVariableOp3Encoder-1st-NormalizationLayer-2/add/ReadVariableOp2j
3Encoder-1st-NormalizationLayer-2/mul/ReadVariableOp3Encoder-1st-NormalizationLayer-2/mul/ReadVariableOp2j
3Encoder-2nd-NormalizationLayer-2/add/ReadVariableOp3Encoder-2nd-NormalizationLayer-2/add/ReadVariableOp2j
3Encoder-2nd-NormalizationLayer-2/mul/ReadVariableOp3Encoder-2nd-NormalizationLayer-2/mul/ReadVariableOp2j
3Encoder-FeedForwardLayer_1_2/BiasAdd/ReadVariableOp3Encoder-FeedForwardLayer_1_2/BiasAdd/ReadVariableOp2n
5Encoder-FeedForwardLayer_1_2/Tensordot/ReadVariableOp5Encoder-FeedForwardLayer_1_2/Tensordot/ReadVariableOp2j
3Encoder-FeedForwardLayer_2_2/BiasAdd/ReadVariableOp3Encoder-FeedForwardLayer_2_2/BiasAdd/ReadVariableOp2n
5Encoder-FeedForwardLayer_2_2/Tensordot/ReadVariableOp5Encoder-FeedForwardLayer_2_2/Tensordot/ReadVariableOp2�
@Encoder-SelfAttentionLayer-2/attention_output/add/ReadVariableOp@Encoder-SelfAttentionLayer-2/attention_output/add/ReadVariableOp2�
JEncoder-SelfAttentionLayer-2/attention_output/einsum/Einsum/ReadVariableOpJEncoder-SelfAttentionLayer-2/attention_output/einsum/Einsum/ReadVariableOp2j
3Encoder-SelfAttentionLayer-2/key/add/ReadVariableOp3Encoder-SelfAttentionLayer-2/key/add/ReadVariableOp2~
=Encoder-SelfAttentionLayer-2/key/einsum/Einsum/ReadVariableOp=Encoder-SelfAttentionLayer-2/key/einsum/Einsum/ReadVariableOp2n
5Encoder-SelfAttentionLayer-2/query/add/ReadVariableOp5Encoder-SelfAttentionLayer-2/query/add/ReadVariableOp2�
?Encoder-SelfAttentionLayer-2/query/einsum/Einsum/ReadVariableOp?Encoder-SelfAttentionLayer-2/query/einsum/Einsum/ReadVariableOp2n
5Encoder-SelfAttentionLayer-2/value/add/ReadVariableOp5Encoder-SelfAttentionLayer-2/value/add/ReadVariableOp2�
?Encoder-SelfAttentionLayer-2/value/einsum/Einsum/ReadVariableOp?Encoder-SelfAttentionLayer-2/value/einsum/Einsum/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:(
$
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:S O
+
_output_shapes
:���������P	
 
_user_specified_nameinputs
� 
�
J__inference_FinalLayerNorm_layer_call_and_return_conditional_losses_233260

inputs)
mul_readvariableop_resource:	)
add_readvariableop_resource:	
identity��add/ReadVariableOp�mul/ReadVariableOpI
ShapeShapeinputs*
T0*
_output_shapes
::��]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskO
ConstConst*
_output_shapes
:*
dtype0*
valueB: U
ProdProdstrided_slice:output:0Const:output:0*
T0*
_output_shapes
: _
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_maskQ
Const_1Const*
_output_shapes
:*
dtype0*
valueB: [
Prod_1Prodstrided_slice_1:output:0Const_1:output:0*
T0*
_output_shapes
: Q
Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :Q
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :�
Reshape/shapePackReshape/shape/0:output:0Prod:output:0Prod_1:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:u
ReshapeReshapeinputsReshape/shape:output:0*
T0*8
_output_shapes&
$:"������������������P
ones/packedPackProd:output:0*
N*
T0*
_output_shapes
:O

ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?e
onesFillones/packed:output:0ones/Const:output:0*
T0*#
_output_shapes
:���������Q
zeros/packedPackProd:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    h
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*#
_output_shapes
:���������J
Const_2Const*
_output_shapes
: *
dtype0*
valueB J
Const_3Const*
_output_shapes
: *
dtype0*
valueB �
FusedBatchNormV3FusedBatchNormV3Reshape:output:0ones:output:0zeros:output:0Const_2:output:0Const_3:output:0*
T0*
U0*x
_output_shapesf
d:"������������������:���������:���������:���������:���������:*
data_formatNCHW*
epsilon%o�:p
	Reshape_1ReshapeFusedBatchNormV3:y:0Shape:output:0*
T0*+
_output_shapes
:���������P	j
mul/ReadVariableOpReadVariableOpmul_readvariableop_resource*
_output_shapes
:	*
dtype0p
mulMulReshape_1:output:0mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������P	j
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
:	*
dtype0g
addAddV2mul:z:0add/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������P	Z
IdentityIdentityadd:z:0^NoOp*
T0*+
_output_shapes
:���������P	L
NoOpNoOp^add/ReadVariableOp^mul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������P	: : 2(
add/ReadVariableOpadd/ReadVariableOp2(
mul/ReadVariableOpmul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:S O
+
_output_shapes
:���������P	
 
_user_specified_nameinputs
�

�
Q__inference_PredictionImprovement_layer_call_and_return_conditional_losses_233372

inputs0
matmul_readvariableop_resource:
-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:���������Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:���������S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
'
_output_shapes
:���������

 
_user_specified_nameinputs
��
�
Q__inference_transformer_encoder_5_layer_call_and_return_conditional_losses_233035

inputsJ
<encoder_1st_normalizationlayer_3_mul_readvariableop_resource:	J
<encoder_1st_normalizationlayer_3_add_readvariableop_resource:	^
Hencoder_selfattentionlayer_3_query_einsum_einsum_readvariableop_resource:	P
>encoder_selfattentionlayer_3_query_add_readvariableop_resource:\
Fencoder_selfattentionlayer_3_key_einsum_einsum_readvariableop_resource:	N
<encoder_selfattentionlayer_3_key_add_readvariableop_resource:^
Hencoder_selfattentionlayer_3_value_einsum_einsum_readvariableop_resource:	P
>encoder_selfattentionlayer_3_value_add_readvariableop_resource:i
Sencoder_selfattentionlayer_3_attention_output_einsum_einsum_readvariableop_resource:	W
Iencoder_selfattentionlayer_3_attention_output_add_readvariableop_resource:	J
<encoder_2nd_normalizationlayer_3_mul_readvariableop_resource:	J
<encoder_2nd_normalizationlayer_3_add_readvariableop_resource:	P
>encoder_feedforwardlayer_1_3_tensordot_readvariableop_resource:		J
<encoder_feedforwardlayer_1_3_biasadd_readvariableop_resource:	P
>encoder_feedforwardlayer_2_3_tensordot_readvariableop_resource:		J
<encoder_feedforwardlayer_2_3_biasadd_readvariableop_resource:	
identity

identity_1��3Encoder-1st-NormalizationLayer-3/add/ReadVariableOp�3Encoder-1st-NormalizationLayer-3/mul/ReadVariableOp�3Encoder-2nd-NormalizationLayer-3/add/ReadVariableOp�3Encoder-2nd-NormalizationLayer-3/mul/ReadVariableOp�3Encoder-FeedForwardLayer_1_3/BiasAdd/ReadVariableOp�5Encoder-FeedForwardLayer_1_3/Tensordot/ReadVariableOp�3Encoder-FeedForwardLayer_2_3/BiasAdd/ReadVariableOp�5Encoder-FeedForwardLayer_2_3/Tensordot/ReadVariableOp�@Encoder-SelfAttentionLayer-3/attention_output/add/ReadVariableOp�JEncoder-SelfAttentionLayer-3/attention_output/einsum/Einsum/ReadVariableOp�3Encoder-SelfAttentionLayer-3/key/add/ReadVariableOp�=Encoder-SelfAttentionLayer-3/key/einsum/Einsum/ReadVariableOp�5Encoder-SelfAttentionLayer-3/query/add/ReadVariableOp�?Encoder-SelfAttentionLayer-3/query/einsum/Einsum/ReadVariableOp�5Encoder-SelfAttentionLayer-3/value/add/ReadVariableOp�?Encoder-SelfAttentionLayer-3/value/einsum/Einsum/ReadVariableOpj
&Encoder-1st-NormalizationLayer-3/ShapeShapeinputs*
T0*
_output_shapes
::��~
4Encoder-1st-NormalizationLayer-3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
6Encoder-1st-NormalizationLayer-3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
6Encoder-1st-NormalizationLayer-3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
.Encoder-1st-NormalizationLayer-3/strided_sliceStridedSlice/Encoder-1st-NormalizationLayer-3/Shape:output:0=Encoder-1st-NormalizationLayer-3/strided_slice/stack:output:0?Encoder-1st-NormalizationLayer-3/strided_slice/stack_1:output:0?Encoder-1st-NormalizationLayer-3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskp
&Encoder-1st-NormalizationLayer-3/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
%Encoder-1st-NormalizationLayer-3/ProdProd7Encoder-1st-NormalizationLayer-3/strided_slice:output:0/Encoder-1st-NormalizationLayer-3/Const:output:0*
T0*
_output_shapes
: �
6Encoder-1st-NormalizationLayer-3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:�
8Encoder-1st-NormalizationLayer-3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: �
8Encoder-1st-NormalizationLayer-3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
0Encoder-1st-NormalizationLayer-3/strided_slice_1StridedSlice/Encoder-1st-NormalizationLayer-3/Shape:output:0?Encoder-1st-NormalizationLayer-3/strided_slice_1/stack:output:0AEncoder-1st-NormalizationLayer-3/strided_slice_1/stack_1:output:0AEncoder-1st-NormalizationLayer-3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_maskr
(Encoder-1st-NormalizationLayer-3/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
'Encoder-1st-NormalizationLayer-3/Prod_1Prod9Encoder-1st-NormalizationLayer-3/strided_slice_1:output:01Encoder-1st-NormalizationLayer-3/Const_1:output:0*
T0*
_output_shapes
: r
0Encoder-1st-NormalizationLayer-3/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :r
0Encoder-1st-NormalizationLayer-3/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :�
.Encoder-1st-NormalizationLayer-3/Reshape/shapePack9Encoder-1st-NormalizationLayer-3/Reshape/shape/0:output:0.Encoder-1st-NormalizationLayer-3/Prod:output:00Encoder-1st-NormalizationLayer-3/Prod_1:output:09Encoder-1st-NormalizationLayer-3/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:�
(Encoder-1st-NormalizationLayer-3/ReshapeReshapeinputs7Encoder-1st-NormalizationLayer-3/Reshape/shape:output:0*
T0*8
_output_shapes&
$:"�������������������
,Encoder-1st-NormalizationLayer-3/ones/packedPack.Encoder-1st-NormalizationLayer-3/Prod:output:0*
N*
T0*
_output_shapes
:p
+Encoder-1st-NormalizationLayer-3/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
%Encoder-1st-NormalizationLayer-3/onesFill5Encoder-1st-NormalizationLayer-3/ones/packed:output:04Encoder-1st-NormalizationLayer-3/ones/Const:output:0*
T0*#
_output_shapes
:����������
-Encoder-1st-NormalizationLayer-3/zeros/packedPack.Encoder-1st-NormalizationLayer-3/Prod:output:0*
N*
T0*
_output_shapes
:q
,Encoder-1st-NormalizationLayer-3/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
&Encoder-1st-NormalizationLayer-3/zerosFill6Encoder-1st-NormalizationLayer-3/zeros/packed:output:05Encoder-1st-NormalizationLayer-3/zeros/Const:output:0*
T0*#
_output_shapes
:���������k
(Encoder-1st-NormalizationLayer-3/Const_2Const*
_output_shapes
: *
dtype0*
valueB k
(Encoder-1st-NormalizationLayer-3/Const_3Const*
_output_shapes
: *
dtype0*
valueB �
1Encoder-1st-NormalizationLayer-3/FusedBatchNormV3FusedBatchNormV31Encoder-1st-NormalizationLayer-3/Reshape:output:0.Encoder-1st-NormalizationLayer-3/ones:output:0/Encoder-1st-NormalizationLayer-3/zeros:output:01Encoder-1st-NormalizationLayer-3/Const_2:output:01Encoder-1st-NormalizationLayer-3/Const_3:output:0*
T0*
U0*x
_output_shapesf
d:"������������������:���������:���������:���������:���������:*
data_formatNCHW*
epsilon%o�:�
*Encoder-1st-NormalizationLayer-3/Reshape_1Reshape5Encoder-1st-NormalizationLayer-3/FusedBatchNormV3:y:0/Encoder-1st-NormalizationLayer-3/Shape:output:0*
T0*+
_output_shapes
:���������P	�
3Encoder-1st-NormalizationLayer-3/mul/ReadVariableOpReadVariableOp<encoder_1st_normalizationlayer_3_mul_readvariableop_resource*
_output_shapes
:	*
dtype0�
$Encoder-1st-NormalizationLayer-3/mulMul3Encoder-1st-NormalizationLayer-3/Reshape_1:output:0;Encoder-1st-NormalizationLayer-3/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������P	�
3Encoder-1st-NormalizationLayer-3/add/ReadVariableOpReadVariableOp<encoder_1st_normalizationlayer_3_add_readvariableop_resource*
_output_shapes
:	*
dtype0�
$Encoder-1st-NormalizationLayer-3/addAddV2(Encoder-1st-NormalizationLayer-3/mul:z:0;Encoder-1st-NormalizationLayer-3/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������P	�
?Encoder-SelfAttentionLayer-3/query/einsum/Einsum/ReadVariableOpReadVariableOpHencoder_selfattentionlayer_3_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:	*
dtype0�
0Encoder-SelfAttentionLayer-3/query/einsum/EinsumEinsum(Encoder-1st-NormalizationLayer-3/add:z:0GEncoder-SelfAttentionLayer-3/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������P*
equationabc,cde->abde�
5Encoder-SelfAttentionLayer-3/query/add/ReadVariableOpReadVariableOp>encoder_selfattentionlayer_3_query_add_readvariableop_resource*
_output_shapes

:*
dtype0�
&Encoder-SelfAttentionLayer-3/query/addAddV29Encoder-SelfAttentionLayer-3/query/einsum/Einsum:output:0=Encoder-SelfAttentionLayer-3/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������P�
=Encoder-SelfAttentionLayer-3/key/einsum/Einsum/ReadVariableOpReadVariableOpFencoder_selfattentionlayer_3_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:	*
dtype0�
.Encoder-SelfAttentionLayer-3/key/einsum/EinsumEinsum(Encoder-1st-NormalizationLayer-3/add:z:0EEncoder-SelfAttentionLayer-3/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������P*
equationabc,cde->abde�
3Encoder-SelfAttentionLayer-3/key/add/ReadVariableOpReadVariableOp<encoder_selfattentionlayer_3_key_add_readvariableop_resource*
_output_shapes

:*
dtype0�
$Encoder-SelfAttentionLayer-3/key/addAddV27Encoder-SelfAttentionLayer-3/key/einsum/Einsum:output:0;Encoder-SelfAttentionLayer-3/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������P�
?Encoder-SelfAttentionLayer-3/value/einsum/Einsum/ReadVariableOpReadVariableOpHencoder_selfattentionlayer_3_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:	*
dtype0�
0Encoder-SelfAttentionLayer-3/value/einsum/EinsumEinsum(Encoder-1st-NormalizationLayer-3/add:z:0GEncoder-SelfAttentionLayer-3/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������P*
equationabc,cde->abde�
5Encoder-SelfAttentionLayer-3/value/add/ReadVariableOpReadVariableOp>encoder_selfattentionlayer_3_value_add_readvariableop_resource*
_output_shapes

:*
dtype0�
&Encoder-SelfAttentionLayer-3/value/addAddV29Encoder-SelfAttentionLayer-3/value/einsum/Einsum:output:0=Encoder-SelfAttentionLayer-3/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������Pg
"Encoder-SelfAttentionLayer-3/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *:�?�
 Encoder-SelfAttentionLayer-3/MulMul*Encoder-SelfAttentionLayer-3/query/add:z:0+Encoder-SelfAttentionLayer-3/Mul/y:output:0*
T0*/
_output_shapes
:���������P�
*Encoder-SelfAttentionLayer-3/einsum/EinsumEinsum(Encoder-SelfAttentionLayer-3/key/add:z:0$Encoder-SelfAttentionLayer-3/Mul:z:0*
N*
T0*/
_output_shapes
:���������PP*
equationaecd,abcd->acbe�
,Encoder-SelfAttentionLayer-3/softmax/SoftmaxSoftmax3Encoder-SelfAttentionLayer-3/einsum/Einsum:output:0*
T0*/
_output_shapes
:���������PPw
2Encoder-SelfAttentionLayer-3/dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?�
0Encoder-SelfAttentionLayer-3/dropout/dropout/MulMul6Encoder-SelfAttentionLayer-3/softmax/Softmax:softmax:0;Encoder-SelfAttentionLayer-3/dropout/dropout/Const:output:0*
T0*/
_output_shapes
:���������PP�
2Encoder-SelfAttentionLayer-3/dropout/dropout/ShapeShape6Encoder-SelfAttentionLayer-3/softmax/Softmax:softmax:0*
T0*
_output_shapes
::���
IEncoder-SelfAttentionLayer-3/dropout/dropout/random_uniform/RandomUniformRandomUniform;Encoder-SelfAttentionLayer-3/dropout/dropout/Shape:output:0*
T0*/
_output_shapes
:���������PP*
dtype0*
seed���
;Encoder-SelfAttentionLayer-3/dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=�
9Encoder-SelfAttentionLayer-3/dropout/dropout/GreaterEqualGreaterEqualREncoder-SelfAttentionLayer-3/dropout/dropout/random_uniform/RandomUniform:output:0DEncoder-SelfAttentionLayer-3/dropout/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:���������PPy
4Encoder-SelfAttentionLayer-3/dropout/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
5Encoder-SelfAttentionLayer-3/dropout/dropout/SelectV2SelectV2=Encoder-SelfAttentionLayer-3/dropout/dropout/GreaterEqual:z:04Encoder-SelfAttentionLayer-3/dropout/dropout/Mul:z:0=Encoder-SelfAttentionLayer-3/dropout/dropout/Const_1:output:0*
T0*/
_output_shapes
:���������PP�
,Encoder-SelfAttentionLayer-3/einsum_1/EinsumEinsum>Encoder-SelfAttentionLayer-3/dropout/dropout/SelectV2:output:0*Encoder-SelfAttentionLayer-3/value/add:z:0*
N*
T0*/
_output_shapes
:���������P*
equationacbe,aecd->abcd�
JEncoder-SelfAttentionLayer-3/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpSencoder_selfattentionlayer_3_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:	*
dtype0�
;Encoder-SelfAttentionLayer-3/attention_output/einsum/EinsumEinsum5Encoder-SelfAttentionLayer-3/einsum_1/Einsum:output:0REncoder-SelfAttentionLayer-3/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:���������P	*
equationabcd,cde->abe�
@Encoder-SelfAttentionLayer-3/attention_output/add/ReadVariableOpReadVariableOpIencoder_selfattentionlayer_3_attention_output_add_readvariableop_resource*
_output_shapes
:	*
dtype0�
1Encoder-SelfAttentionLayer-3/attention_output/addAddV2DEncoder-SelfAttentionLayer-3/attention_output/einsum/Einsum:output:0HEncoder-SelfAttentionLayer-3/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������P	�
Encoder-1st-AdditionLayer-3/addAddV2inputs5Encoder-SelfAttentionLayer-3/attention_output/add:z:0*
T0*+
_output_shapes
:���������P	�
&Encoder-2nd-NormalizationLayer-3/ShapeShape#Encoder-1st-AdditionLayer-3/add:z:0*
T0*
_output_shapes
::��~
4Encoder-2nd-NormalizationLayer-3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
6Encoder-2nd-NormalizationLayer-3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
6Encoder-2nd-NormalizationLayer-3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
.Encoder-2nd-NormalizationLayer-3/strided_sliceStridedSlice/Encoder-2nd-NormalizationLayer-3/Shape:output:0=Encoder-2nd-NormalizationLayer-3/strided_slice/stack:output:0?Encoder-2nd-NormalizationLayer-3/strided_slice/stack_1:output:0?Encoder-2nd-NormalizationLayer-3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskp
&Encoder-2nd-NormalizationLayer-3/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
%Encoder-2nd-NormalizationLayer-3/ProdProd7Encoder-2nd-NormalizationLayer-3/strided_slice:output:0/Encoder-2nd-NormalizationLayer-3/Const:output:0*
T0*
_output_shapes
: �
6Encoder-2nd-NormalizationLayer-3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:�
8Encoder-2nd-NormalizationLayer-3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: �
8Encoder-2nd-NormalizationLayer-3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
0Encoder-2nd-NormalizationLayer-3/strided_slice_1StridedSlice/Encoder-2nd-NormalizationLayer-3/Shape:output:0?Encoder-2nd-NormalizationLayer-3/strided_slice_1/stack:output:0AEncoder-2nd-NormalizationLayer-3/strided_slice_1/stack_1:output:0AEncoder-2nd-NormalizationLayer-3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_maskr
(Encoder-2nd-NormalizationLayer-3/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
'Encoder-2nd-NormalizationLayer-3/Prod_1Prod9Encoder-2nd-NormalizationLayer-3/strided_slice_1:output:01Encoder-2nd-NormalizationLayer-3/Const_1:output:0*
T0*
_output_shapes
: r
0Encoder-2nd-NormalizationLayer-3/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :r
0Encoder-2nd-NormalizationLayer-3/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :�
.Encoder-2nd-NormalizationLayer-3/Reshape/shapePack9Encoder-2nd-NormalizationLayer-3/Reshape/shape/0:output:0.Encoder-2nd-NormalizationLayer-3/Prod:output:00Encoder-2nd-NormalizationLayer-3/Prod_1:output:09Encoder-2nd-NormalizationLayer-3/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:�
(Encoder-2nd-NormalizationLayer-3/ReshapeReshape#Encoder-1st-AdditionLayer-3/add:z:07Encoder-2nd-NormalizationLayer-3/Reshape/shape:output:0*
T0*8
_output_shapes&
$:"�������������������
,Encoder-2nd-NormalizationLayer-3/ones/packedPack.Encoder-2nd-NormalizationLayer-3/Prod:output:0*
N*
T0*
_output_shapes
:p
+Encoder-2nd-NormalizationLayer-3/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
%Encoder-2nd-NormalizationLayer-3/onesFill5Encoder-2nd-NormalizationLayer-3/ones/packed:output:04Encoder-2nd-NormalizationLayer-3/ones/Const:output:0*
T0*#
_output_shapes
:����������
-Encoder-2nd-NormalizationLayer-3/zeros/packedPack.Encoder-2nd-NormalizationLayer-3/Prod:output:0*
N*
T0*
_output_shapes
:q
,Encoder-2nd-NormalizationLayer-3/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
&Encoder-2nd-NormalizationLayer-3/zerosFill6Encoder-2nd-NormalizationLayer-3/zeros/packed:output:05Encoder-2nd-NormalizationLayer-3/zeros/Const:output:0*
T0*#
_output_shapes
:���������k
(Encoder-2nd-NormalizationLayer-3/Const_2Const*
_output_shapes
: *
dtype0*
valueB k
(Encoder-2nd-NormalizationLayer-3/Const_3Const*
_output_shapes
: *
dtype0*
valueB �
1Encoder-2nd-NormalizationLayer-3/FusedBatchNormV3FusedBatchNormV31Encoder-2nd-NormalizationLayer-3/Reshape:output:0.Encoder-2nd-NormalizationLayer-3/ones:output:0/Encoder-2nd-NormalizationLayer-3/zeros:output:01Encoder-2nd-NormalizationLayer-3/Const_2:output:01Encoder-2nd-NormalizationLayer-3/Const_3:output:0*
T0*
U0*x
_output_shapesf
d:"������������������:���������:���������:���������:���������:*
data_formatNCHW*
epsilon%o�:�
*Encoder-2nd-NormalizationLayer-3/Reshape_1Reshape5Encoder-2nd-NormalizationLayer-3/FusedBatchNormV3:y:0/Encoder-2nd-NormalizationLayer-3/Shape:output:0*
T0*+
_output_shapes
:���������P	�
3Encoder-2nd-NormalizationLayer-3/mul/ReadVariableOpReadVariableOp<encoder_2nd_normalizationlayer_3_mul_readvariableop_resource*
_output_shapes
:	*
dtype0�
$Encoder-2nd-NormalizationLayer-3/mulMul3Encoder-2nd-NormalizationLayer-3/Reshape_1:output:0;Encoder-2nd-NormalizationLayer-3/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������P	�
3Encoder-2nd-NormalizationLayer-3/add/ReadVariableOpReadVariableOp<encoder_2nd_normalizationlayer_3_add_readvariableop_resource*
_output_shapes
:	*
dtype0�
$Encoder-2nd-NormalizationLayer-3/addAddV2(Encoder-2nd-NormalizationLayer-3/mul:z:0;Encoder-2nd-NormalizationLayer-3/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������P	�
5Encoder-FeedForwardLayer_1_3/Tensordot/ReadVariableOpReadVariableOp>encoder_feedforwardlayer_1_3_tensordot_readvariableop_resource*
_output_shapes

:		*
dtype0u
+Encoder-FeedForwardLayer_1_3/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:|
+Encoder-FeedForwardLayer_1_3/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       �
,Encoder-FeedForwardLayer_1_3/Tensordot/ShapeShape(Encoder-2nd-NormalizationLayer-3/add:z:0*
T0*
_output_shapes
::��v
4Encoder-FeedForwardLayer_1_3/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
/Encoder-FeedForwardLayer_1_3/Tensordot/GatherV2GatherV25Encoder-FeedForwardLayer_1_3/Tensordot/Shape:output:04Encoder-FeedForwardLayer_1_3/Tensordot/free:output:0=Encoder-FeedForwardLayer_1_3/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:x
6Encoder-FeedForwardLayer_1_3/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
1Encoder-FeedForwardLayer_1_3/Tensordot/GatherV2_1GatherV25Encoder-FeedForwardLayer_1_3/Tensordot/Shape:output:04Encoder-FeedForwardLayer_1_3/Tensordot/axes:output:0?Encoder-FeedForwardLayer_1_3/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:v
,Encoder-FeedForwardLayer_1_3/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
+Encoder-FeedForwardLayer_1_3/Tensordot/ProdProd8Encoder-FeedForwardLayer_1_3/Tensordot/GatherV2:output:05Encoder-FeedForwardLayer_1_3/Tensordot/Const:output:0*
T0*
_output_shapes
: x
.Encoder-FeedForwardLayer_1_3/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
-Encoder-FeedForwardLayer_1_3/Tensordot/Prod_1Prod:Encoder-FeedForwardLayer_1_3/Tensordot/GatherV2_1:output:07Encoder-FeedForwardLayer_1_3/Tensordot/Const_1:output:0*
T0*
_output_shapes
: t
2Encoder-FeedForwardLayer_1_3/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
-Encoder-FeedForwardLayer_1_3/Tensordot/concatConcatV24Encoder-FeedForwardLayer_1_3/Tensordot/free:output:04Encoder-FeedForwardLayer_1_3/Tensordot/axes:output:0;Encoder-FeedForwardLayer_1_3/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
,Encoder-FeedForwardLayer_1_3/Tensordot/stackPack4Encoder-FeedForwardLayer_1_3/Tensordot/Prod:output:06Encoder-FeedForwardLayer_1_3/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
0Encoder-FeedForwardLayer_1_3/Tensordot/transpose	Transpose(Encoder-2nd-NormalizationLayer-3/add:z:06Encoder-FeedForwardLayer_1_3/Tensordot/concat:output:0*
T0*+
_output_shapes
:���������P	�
.Encoder-FeedForwardLayer_1_3/Tensordot/ReshapeReshape4Encoder-FeedForwardLayer_1_3/Tensordot/transpose:y:05Encoder-FeedForwardLayer_1_3/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
-Encoder-FeedForwardLayer_1_3/Tensordot/MatMulMatMul7Encoder-FeedForwardLayer_1_3/Tensordot/Reshape:output:0=Encoder-FeedForwardLayer_1_3/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������	x
.Encoder-FeedForwardLayer_1_3/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:	v
4Encoder-FeedForwardLayer_1_3/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
/Encoder-FeedForwardLayer_1_3/Tensordot/concat_1ConcatV28Encoder-FeedForwardLayer_1_3/Tensordot/GatherV2:output:07Encoder-FeedForwardLayer_1_3/Tensordot/Const_2:output:0=Encoder-FeedForwardLayer_1_3/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
&Encoder-FeedForwardLayer_1_3/TensordotReshape7Encoder-FeedForwardLayer_1_3/Tensordot/MatMul:product:08Encoder-FeedForwardLayer_1_3/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������P	�
3Encoder-FeedForwardLayer_1_3/BiasAdd/ReadVariableOpReadVariableOp<encoder_feedforwardlayer_1_3_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype0�
$Encoder-FeedForwardLayer_1_3/BiasAddBiasAdd/Encoder-FeedForwardLayer_1_3/Tensordot:output:0;Encoder-FeedForwardLayer_1_3/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������P	l
'Encoder-FeedForwardLayer_1_3/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
%Encoder-FeedForwardLayer_1_3/Gelu/mulMul0Encoder-FeedForwardLayer_1_3/Gelu/mul/x:output:0-Encoder-FeedForwardLayer_1_3/BiasAdd:output:0*
T0*+
_output_shapes
:���������P	m
(Encoder-FeedForwardLayer_1_3/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *��?�
)Encoder-FeedForwardLayer_1_3/Gelu/truedivRealDiv-Encoder-FeedForwardLayer_1_3/BiasAdd:output:01Encoder-FeedForwardLayer_1_3/Gelu/Cast/x:output:0*
T0*+
_output_shapes
:���������P	�
%Encoder-FeedForwardLayer_1_3/Gelu/ErfErf-Encoder-FeedForwardLayer_1_3/Gelu/truediv:z:0*
T0*+
_output_shapes
:���������P	l
'Encoder-FeedForwardLayer_1_3/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
%Encoder-FeedForwardLayer_1_3/Gelu/addAddV20Encoder-FeedForwardLayer_1_3/Gelu/add/x:output:0)Encoder-FeedForwardLayer_1_3/Gelu/Erf:y:0*
T0*+
_output_shapes
:���������P	�
'Encoder-FeedForwardLayer_1_3/Gelu/mul_1Mul)Encoder-FeedForwardLayer_1_3/Gelu/mul:z:0)Encoder-FeedForwardLayer_1_3/Gelu/add:z:0*
T0*+
_output_shapes
:���������P	�
5Encoder-FeedForwardLayer_2_3/Tensordot/ReadVariableOpReadVariableOp>encoder_feedforwardlayer_2_3_tensordot_readvariableop_resource*
_output_shapes

:		*
dtype0u
+Encoder-FeedForwardLayer_2_3/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:|
+Encoder-FeedForwardLayer_2_3/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       �
,Encoder-FeedForwardLayer_2_3/Tensordot/ShapeShape+Encoder-FeedForwardLayer_1_3/Gelu/mul_1:z:0*
T0*
_output_shapes
::��v
4Encoder-FeedForwardLayer_2_3/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
/Encoder-FeedForwardLayer_2_3/Tensordot/GatherV2GatherV25Encoder-FeedForwardLayer_2_3/Tensordot/Shape:output:04Encoder-FeedForwardLayer_2_3/Tensordot/free:output:0=Encoder-FeedForwardLayer_2_3/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:x
6Encoder-FeedForwardLayer_2_3/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
1Encoder-FeedForwardLayer_2_3/Tensordot/GatherV2_1GatherV25Encoder-FeedForwardLayer_2_3/Tensordot/Shape:output:04Encoder-FeedForwardLayer_2_3/Tensordot/axes:output:0?Encoder-FeedForwardLayer_2_3/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:v
,Encoder-FeedForwardLayer_2_3/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
+Encoder-FeedForwardLayer_2_3/Tensordot/ProdProd8Encoder-FeedForwardLayer_2_3/Tensordot/GatherV2:output:05Encoder-FeedForwardLayer_2_3/Tensordot/Const:output:0*
T0*
_output_shapes
: x
.Encoder-FeedForwardLayer_2_3/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
-Encoder-FeedForwardLayer_2_3/Tensordot/Prod_1Prod:Encoder-FeedForwardLayer_2_3/Tensordot/GatherV2_1:output:07Encoder-FeedForwardLayer_2_3/Tensordot/Const_1:output:0*
T0*
_output_shapes
: t
2Encoder-FeedForwardLayer_2_3/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
-Encoder-FeedForwardLayer_2_3/Tensordot/concatConcatV24Encoder-FeedForwardLayer_2_3/Tensordot/free:output:04Encoder-FeedForwardLayer_2_3/Tensordot/axes:output:0;Encoder-FeedForwardLayer_2_3/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
,Encoder-FeedForwardLayer_2_3/Tensordot/stackPack4Encoder-FeedForwardLayer_2_3/Tensordot/Prod:output:06Encoder-FeedForwardLayer_2_3/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
0Encoder-FeedForwardLayer_2_3/Tensordot/transpose	Transpose+Encoder-FeedForwardLayer_1_3/Gelu/mul_1:z:06Encoder-FeedForwardLayer_2_3/Tensordot/concat:output:0*
T0*+
_output_shapes
:���������P	�
.Encoder-FeedForwardLayer_2_3/Tensordot/ReshapeReshape4Encoder-FeedForwardLayer_2_3/Tensordot/transpose:y:05Encoder-FeedForwardLayer_2_3/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
-Encoder-FeedForwardLayer_2_3/Tensordot/MatMulMatMul7Encoder-FeedForwardLayer_2_3/Tensordot/Reshape:output:0=Encoder-FeedForwardLayer_2_3/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������	x
.Encoder-FeedForwardLayer_2_3/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:	v
4Encoder-FeedForwardLayer_2_3/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
/Encoder-FeedForwardLayer_2_3/Tensordot/concat_1ConcatV28Encoder-FeedForwardLayer_2_3/Tensordot/GatherV2:output:07Encoder-FeedForwardLayer_2_3/Tensordot/Const_2:output:0=Encoder-FeedForwardLayer_2_3/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
&Encoder-FeedForwardLayer_2_3/TensordotReshape7Encoder-FeedForwardLayer_2_3/Tensordot/MatMul:product:08Encoder-FeedForwardLayer_2_3/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������P	�
3Encoder-FeedForwardLayer_2_3/BiasAdd/ReadVariableOpReadVariableOp<encoder_feedforwardlayer_2_3_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype0�
$Encoder-FeedForwardLayer_2_3/BiasAddBiasAdd/Encoder-FeedForwardLayer_2_3/Tensordot:output:0;Encoder-FeedForwardLayer_2_3/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������P	\
dropout_5/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?�
dropout_5/dropout/MulMul-Encoder-FeedForwardLayer_2_3/BiasAdd:output:0 dropout_5/dropout/Const:output:0*
T0*+
_output_shapes
:���������P	�
dropout_5/dropout/ShapeShape-Encoder-FeedForwardLayer_2_3/BiasAdd:output:0*
T0*
_output_shapes
::���
.dropout_5/dropout/random_uniform/RandomUniformRandomUniform dropout_5/dropout/Shape:output:0*
T0*+
_output_shapes
:���������P	*
dtype0*
seed2*
seed��e
 dropout_5/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=�
dropout_5/dropout/GreaterEqualGreaterEqual7dropout_5/dropout/random_uniform/RandomUniform:output:0)dropout_5/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:���������P	^
dropout_5/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout_5/dropout/SelectV2SelectV2"dropout_5/dropout/GreaterEqual:z:0dropout_5/dropout/Mul:z:0"dropout_5/dropout/Const_1:output:0*
T0*+
_output_shapes
:���������P	�
Encoder-2nd-AdditionLayer-3/addAddV2#Encoder-1st-AdditionLayer-3/add:z:0#dropout_5/dropout/SelectV2:output:0*
T0*+
_output_shapes
:���������P	v
IdentityIdentity#Encoder-2nd-AdditionLayer-3/add:z:0^NoOp*
T0*+
_output_shapes
:���������P	�

Identity_1Identity6Encoder-SelfAttentionLayer-3/softmax/Softmax:softmax:0^NoOp*
T0*/
_output_shapes
:���������PP�
NoOpNoOp4^Encoder-1st-NormalizationLayer-3/add/ReadVariableOp4^Encoder-1st-NormalizationLayer-3/mul/ReadVariableOp4^Encoder-2nd-NormalizationLayer-3/add/ReadVariableOp4^Encoder-2nd-NormalizationLayer-3/mul/ReadVariableOp4^Encoder-FeedForwardLayer_1_3/BiasAdd/ReadVariableOp6^Encoder-FeedForwardLayer_1_3/Tensordot/ReadVariableOp4^Encoder-FeedForwardLayer_2_3/BiasAdd/ReadVariableOp6^Encoder-FeedForwardLayer_2_3/Tensordot/ReadVariableOpA^Encoder-SelfAttentionLayer-3/attention_output/add/ReadVariableOpK^Encoder-SelfAttentionLayer-3/attention_output/einsum/Einsum/ReadVariableOp4^Encoder-SelfAttentionLayer-3/key/add/ReadVariableOp>^Encoder-SelfAttentionLayer-3/key/einsum/Einsum/ReadVariableOp6^Encoder-SelfAttentionLayer-3/query/add/ReadVariableOp@^Encoder-SelfAttentionLayer-3/query/einsum/Einsum/ReadVariableOp6^Encoder-SelfAttentionLayer-3/value/add/ReadVariableOp@^Encoder-SelfAttentionLayer-3/value/einsum/Einsum/ReadVariableOp*
_output_shapes
 "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:���������P	: : : : : : : : : : : : : : : : 2j
3Encoder-1st-NormalizationLayer-3/add/ReadVariableOp3Encoder-1st-NormalizationLayer-3/add/ReadVariableOp2j
3Encoder-1st-NormalizationLayer-3/mul/ReadVariableOp3Encoder-1st-NormalizationLayer-3/mul/ReadVariableOp2j
3Encoder-2nd-NormalizationLayer-3/add/ReadVariableOp3Encoder-2nd-NormalizationLayer-3/add/ReadVariableOp2j
3Encoder-2nd-NormalizationLayer-3/mul/ReadVariableOp3Encoder-2nd-NormalizationLayer-3/mul/ReadVariableOp2j
3Encoder-FeedForwardLayer_1_3/BiasAdd/ReadVariableOp3Encoder-FeedForwardLayer_1_3/BiasAdd/ReadVariableOp2n
5Encoder-FeedForwardLayer_1_3/Tensordot/ReadVariableOp5Encoder-FeedForwardLayer_1_3/Tensordot/ReadVariableOp2j
3Encoder-FeedForwardLayer_2_3/BiasAdd/ReadVariableOp3Encoder-FeedForwardLayer_2_3/BiasAdd/ReadVariableOp2n
5Encoder-FeedForwardLayer_2_3/Tensordot/ReadVariableOp5Encoder-FeedForwardLayer_2_3/Tensordot/ReadVariableOp2�
@Encoder-SelfAttentionLayer-3/attention_output/add/ReadVariableOp@Encoder-SelfAttentionLayer-3/attention_output/add/ReadVariableOp2�
JEncoder-SelfAttentionLayer-3/attention_output/einsum/Einsum/ReadVariableOpJEncoder-SelfAttentionLayer-3/attention_output/einsum/Einsum/ReadVariableOp2j
3Encoder-SelfAttentionLayer-3/key/add/ReadVariableOp3Encoder-SelfAttentionLayer-3/key/add/ReadVariableOp2~
=Encoder-SelfAttentionLayer-3/key/einsum/Einsum/ReadVariableOp=Encoder-SelfAttentionLayer-3/key/einsum/Einsum/ReadVariableOp2n
5Encoder-SelfAttentionLayer-3/query/add/ReadVariableOp5Encoder-SelfAttentionLayer-3/query/add/ReadVariableOp2�
?Encoder-SelfAttentionLayer-3/query/einsum/Einsum/ReadVariableOp?Encoder-SelfAttentionLayer-3/query/einsum/Einsum/ReadVariableOp2n
5Encoder-SelfAttentionLayer-3/value/add/ReadVariableOp5Encoder-SelfAttentionLayer-3/value/add/ReadVariableOp2�
?Encoder-SelfAttentionLayer-3/value/einsum/Einsum/ReadVariableOp?Encoder-SelfAttentionLayer-3/value/einsum/Einsum/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:(
$
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:S O
+
_output_shapes
:���������P	
 
_user_specified_nameinputs
�
�
6__inference_transformer_encoder_3_layer_call_fn_231928

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:
	unknown_3:	
	unknown_4:
	unknown_5:	
	unknown_6:
	unknown_7:	
	unknown_8:	
	unknown_9:	

unknown_10:	

unknown_11:		

unknown_12:	

unknown_13:		

unknown_14:	
identity

identity_1��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *F
_output_shapes4
2:���������P	:���������PP*2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *Z
fURS
Q__inference_transformer_encoder_3_layer_call_and_return_conditional_losses_229957s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������P	y

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*/
_output_shapes
:���������PP<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:���������P	: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name231922:&"
 
_user_specified_name231920:&"
 
_user_specified_name231918:&"
 
_user_specified_name231916:&"
 
_user_specified_name231914:&"
 
_user_specified_name231912:&
"
 
_user_specified_name231910:&	"
 
_user_specified_name231908:&"
 
_user_specified_name231906:&"
 
_user_specified_name231904:&"
 
_user_specified_name231902:&"
 
_user_specified_name231900:&"
 
_user_specified_name231898:&"
 
_user_specified_name231896:&"
 
_user_specified_name231894:&"
 
_user_specified_name231892:S O
+
_output_shapes
:���������P	
 
_user_specified_nameinputs
�
l
P__inference_StandardizeTimeLimit_layer_call_and_return_conditional_losses_233312

inputs
identityJ
sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  pAT
subSubinputssub/y:output:0*
T0*'
_output_shapes
:���������N
	truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �@a
truedivRealDivsub:z:0truediv/y:output:0*
T0*'
_output_shapes
:���������S
IdentityIdentitytruediv:z:0*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
d
H__inference_MaskingLayer_layer_call_and_return_conditional_losses_231889

inputs
identityL

NotEqual/yConst*
_output_shapes
: *
dtype0*
value	B j g
NotEqualNotEqualinputsNotEqual/y:output:0*
T0*+
_output_shapes
:���������P	`
Any/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
���������v
AnyAnyNotEqual:z:0Any/reduction_indices:output:0*+
_output_shapes
:���������P*
	keep_dims(_
CastCastAny:output:0*

DstT0*

SrcT0
*+
_output_shapes
:���������PR
mulMulinputsCast:y:0*
T0*+
_output_shapes
:���������P	r
SqueezeSqueezeAny:output:0*
T0
*'
_output_shapes
:���������P*
squeeze_dims

���������S
IdentityIdentitymul:z:0*
T0*+
_output_shapes
:���������P	"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������P	:S O
+
_output_shapes
:���������P	
 
_user_specified_nameinputs
�
d
H__inference_MaskingLayer_layer_call_and_return_conditional_losses_229767

inputs
identityL

NotEqual/yConst*
_output_shapes
: *
dtype0*
value	B j g
NotEqualNotEqualinputsNotEqual/y:output:0*
T0*+
_output_shapes
:���������P	`
Any/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
���������v
AnyAnyNotEqual:z:0Any/reduction_indices:output:0*+
_output_shapes
:���������P*
	keep_dims(_
CastCastAny:output:0*

DstT0*

SrcT0
*+
_output_shapes
:���������PR
mulMulinputsCast:y:0*
T0*+
_output_shapes
:���������P	r
SqueezeSqueezeAny:output:0*
T0
*'
_output_shapes
:���������P*
squeeze_dims

���������S
IdentityIdentitymul:z:0*
T0*+
_output_shapes
:���������P	"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������P	:S O
+
_output_shapes
:���������P	
 
_user_specified_nameinputs
��
�
Q__inference_transformer_encoder_4_layer_call_and_return_conditional_losses_230179

inputsJ
<encoder_1st_normalizationlayer_2_mul_readvariableop_resource:	J
<encoder_1st_normalizationlayer_2_add_readvariableop_resource:	^
Hencoder_selfattentionlayer_2_query_einsum_einsum_readvariableop_resource:	P
>encoder_selfattentionlayer_2_query_add_readvariableop_resource:\
Fencoder_selfattentionlayer_2_key_einsum_einsum_readvariableop_resource:	N
<encoder_selfattentionlayer_2_key_add_readvariableop_resource:^
Hencoder_selfattentionlayer_2_value_einsum_einsum_readvariableop_resource:	P
>encoder_selfattentionlayer_2_value_add_readvariableop_resource:i
Sencoder_selfattentionlayer_2_attention_output_einsum_einsum_readvariableop_resource:	W
Iencoder_selfattentionlayer_2_attention_output_add_readvariableop_resource:	J
<encoder_2nd_normalizationlayer_2_mul_readvariableop_resource:	J
<encoder_2nd_normalizationlayer_2_add_readvariableop_resource:	P
>encoder_feedforwardlayer_1_2_tensordot_readvariableop_resource:		J
<encoder_feedforwardlayer_1_2_biasadd_readvariableop_resource:	P
>encoder_feedforwardlayer_2_2_tensordot_readvariableop_resource:		J
<encoder_feedforwardlayer_2_2_biasadd_readvariableop_resource:	
identity

identity_1��3Encoder-1st-NormalizationLayer-2/add/ReadVariableOp�3Encoder-1st-NormalizationLayer-2/mul/ReadVariableOp�3Encoder-2nd-NormalizationLayer-2/add/ReadVariableOp�3Encoder-2nd-NormalizationLayer-2/mul/ReadVariableOp�3Encoder-FeedForwardLayer_1_2/BiasAdd/ReadVariableOp�5Encoder-FeedForwardLayer_1_2/Tensordot/ReadVariableOp�3Encoder-FeedForwardLayer_2_2/BiasAdd/ReadVariableOp�5Encoder-FeedForwardLayer_2_2/Tensordot/ReadVariableOp�@Encoder-SelfAttentionLayer-2/attention_output/add/ReadVariableOp�JEncoder-SelfAttentionLayer-2/attention_output/einsum/Einsum/ReadVariableOp�3Encoder-SelfAttentionLayer-2/key/add/ReadVariableOp�=Encoder-SelfAttentionLayer-2/key/einsum/Einsum/ReadVariableOp�5Encoder-SelfAttentionLayer-2/query/add/ReadVariableOp�?Encoder-SelfAttentionLayer-2/query/einsum/Einsum/ReadVariableOp�5Encoder-SelfAttentionLayer-2/value/add/ReadVariableOp�?Encoder-SelfAttentionLayer-2/value/einsum/Einsum/ReadVariableOpj
&Encoder-1st-NormalizationLayer-2/ShapeShapeinputs*
T0*
_output_shapes
::��~
4Encoder-1st-NormalizationLayer-2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
6Encoder-1st-NormalizationLayer-2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
6Encoder-1st-NormalizationLayer-2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
.Encoder-1st-NormalizationLayer-2/strided_sliceStridedSlice/Encoder-1st-NormalizationLayer-2/Shape:output:0=Encoder-1st-NormalizationLayer-2/strided_slice/stack:output:0?Encoder-1st-NormalizationLayer-2/strided_slice/stack_1:output:0?Encoder-1st-NormalizationLayer-2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskp
&Encoder-1st-NormalizationLayer-2/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
%Encoder-1st-NormalizationLayer-2/ProdProd7Encoder-1st-NormalizationLayer-2/strided_slice:output:0/Encoder-1st-NormalizationLayer-2/Const:output:0*
T0*
_output_shapes
: �
6Encoder-1st-NormalizationLayer-2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:�
8Encoder-1st-NormalizationLayer-2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: �
8Encoder-1st-NormalizationLayer-2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
0Encoder-1st-NormalizationLayer-2/strided_slice_1StridedSlice/Encoder-1st-NormalizationLayer-2/Shape:output:0?Encoder-1st-NormalizationLayer-2/strided_slice_1/stack:output:0AEncoder-1st-NormalizationLayer-2/strided_slice_1/stack_1:output:0AEncoder-1st-NormalizationLayer-2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_maskr
(Encoder-1st-NormalizationLayer-2/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
'Encoder-1st-NormalizationLayer-2/Prod_1Prod9Encoder-1st-NormalizationLayer-2/strided_slice_1:output:01Encoder-1st-NormalizationLayer-2/Const_1:output:0*
T0*
_output_shapes
: r
0Encoder-1st-NormalizationLayer-2/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :r
0Encoder-1st-NormalizationLayer-2/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :�
.Encoder-1st-NormalizationLayer-2/Reshape/shapePack9Encoder-1st-NormalizationLayer-2/Reshape/shape/0:output:0.Encoder-1st-NormalizationLayer-2/Prod:output:00Encoder-1st-NormalizationLayer-2/Prod_1:output:09Encoder-1st-NormalizationLayer-2/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:�
(Encoder-1st-NormalizationLayer-2/ReshapeReshapeinputs7Encoder-1st-NormalizationLayer-2/Reshape/shape:output:0*
T0*8
_output_shapes&
$:"�������������������
,Encoder-1st-NormalizationLayer-2/ones/packedPack.Encoder-1st-NormalizationLayer-2/Prod:output:0*
N*
T0*
_output_shapes
:p
+Encoder-1st-NormalizationLayer-2/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
%Encoder-1st-NormalizationLayer-2/onesFill5Encoder-1st-NormalizationLayer-2/ones/packed:output:04Encoder-1st-NormalizationLayer-2/ones/Const:output:0*
T0*#
_output_shapes
:����������
-Encoder-1st-NormalizationLayer-2/zeros/packedPack.Encoder-1st-NormalizationLayer-2/Prod:output:0*
N*
T0*
_output_shapes
:q
,Encoder-1st-NormalizationLayer-2/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
&Encoder-1st-NormalizationLayer-2/zerosFill6Encoder-1st-NormalizationLayer-2/zeros/packed:output:05Encoder-1st-NormalizationLayer-2/zeros/Const:output:0*
T0*#
_output_shapes
:���������k
(Encoder-1st-NormalizationLayer-2/Const_2Const*
_output_shapes
: *
dtype0*
valueB k
(Encoder-1st-NormalizationLayer-2/Const_3Const*
_output_shapes
: *
dtype0*
valueB �
1Encoder-1st-NormalizationLayer-2/FusedBatchNormV3FusedBatchNormV31Encoder-1st-NormalizationLayer-2/Reshape:output:0.Encoder-1st-NormalizationLayer-2/ones:output:0/Encoder-1st-NormalizationLayer-2/zeros:output:01Encoder-1st-NormalizationLayer-2/Const_2:output:01Encoder-1st-NormalizationLayer-2/Const_3:output:0*
T0*
U0*x
_output_shapesf
d:"������������������:���������:���������:���������:���������:*
data_formatNCHW*
epsilon%o�:�
*Encoder-1st-NormalizationLayer-2/Reshape_1Reshape5Encoder-1st-NormalizationLayer-2/FusedBatchNormV3:y:0/Encoder-1st-NormalizationLayer-2/Shape:output:0*
T0*+
_output_shapes
:���������P	�
3Encoder-1st-NormalizationLayer-2/mul/ReadVariableOpReadVariableOp<encoder_1st_normalizationlayer_2_mul_readvariableop_resource*
_output_shapes
:	*
dtype0�
$Encoder-1st-NormalizationLayer-2/mulMul3Encoder-1st-NormalizationLayer-2/Reshape_1:output:0;Encoder-1st-NormalizationLayer-2/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������P	�
3Encoder-1st-NormalizationLayer-2/add/ReadVariableOpReadVariableOp<encoder_1st_normalizationlayer_2_add_readvariableop_resource*
_output_shapes
:	*
dtype0�
$Encoder-1st-NormalizationLayer-2/addAddV2(Encoder-1st-NormalizationLayer-2/mul:z:0;Encoder-1st-NormalizationLayer-2/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������P	�
?Encoder-SelfAttentionLayer-2/query/einsum/Einsum/ReadVariableOpReadVariableOpHencoder_selfattentionlayer_2_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:	*
dtype0�
0Encoder-SelfAttentionLayer-2/query/einsum/EinsumEinsum(Encoder-1st-NormalizationLayer-2/add:z:0GEncoder-SelfAttentionLayer-2/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������P*
equationabc,cde->abde�
5Encoder-SelfAttentionLayer-2/query/add/ReadVariableOpReadVariableOp>encoder_selfattentionlayer_2_query_add_readvariableop_resource*
_output_shapes

:*
dtype0�
&Encoder-SelfAttentionLayer-2/query/addAddV29Encoder-SelfAttentionLayer-2/query/einsum/Einsum:output:0=Encoder-SelfAttentionLayer-2/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������P�
=Encoder-SelfAttentionLayer-2/key/einsum/Einsum/ReadVariableOpReadVariableOpFencoder_selfattentionlayer_2_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:	*
dtype0�
.Encoder-SelfAttentionLayer-2/key/einsum/EinsumEinsum(Encoder-1st-NormalizationLayer-2/add:z:0EEncoder-SelfAttentionLayer-2/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������P*
equationabc,cde->abde�
3Encoder-SelfAttentionLayer-2/key/add/ReadVariableOpReadVariableOp<encoder_selfattentionlayer_2_key_add_readvariableop_resource*
_output_shapes

:*
dtype0�
$Encoder-SelfAttentionLayer-2/key/addAddV27Encoder-SelfAttentionLayer-2/key/einsum/Einsum:output:0;Encoder-SelfAttentionLayer-2/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������P�
?Encoder-SelfAttentionLayer-2/value/einsum/Einsum/ReadVariableOpReadVariableOpHencoder_selfattentionlayer_2_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:	*
dtype0�
0Encoder-SelfAttentionLayer-2/value/einsum/EinsumEinsum(Encoder-1st-NormalizationLayer-2/add:z:0GEncoder-SelfAttentionLayer-2/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������P*
equationabc,cde->abde�
5Encoder-SelfAttentionLayer-2/value/add/ReadVariableOpReadVariableOp>encoder_selfattentionlayer_2_value_add_readvariableop_resource*
_output_shapes

:*
dtype0�
&Encoder-SelfAttentionLayer-2/value/addAddV29Encoder-SelfAttentionLayer-2/value/einsum/Einsum:output:0=Encoder-SelfAttentionLayer-2/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������Pg
"Encoder-SelfAttentionLayer-2/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *:�?�
 Encoder-SelfAttentionLayer-2/MulMul*Encoder-SelfAttentionLayer-2/query/add:z:0+Encoder-SelfAttentionLayer-2/Mul/y:output:0*
T0*/
_output_shapes
:���������P�
*Encoder-SelfAttentionLayer-2/einsum/EinsumEinsum(Encoder-SelfAttentionLayer-2/key/add:z:0$Encoder-SelfAttentionLayer-2/Mul:z:0*
N*
T0*/
_output_shapes
:���������PP*
equationaecd,abcd->acbe�
,Encoder-SelfAttentionLayer-2/softmax/SoftmaxSoftmax3Encoder-SelfAttentionLayer-2/einsum/Einsum:output:0*
T0*/
_output_shapes
:���������PPw
2Encoder-SelfAttentionLayer-2/dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?�
0Encoder-SelfAttentionLayer-2/dropout/dropout/MulMul6Encoder-SelfAttentionLayer-2/softmax/Softmax:softmax:0;Encoder-SelfAttentionLayer-2/dropout/dropout/Const:output:0*
T0*/
_output_shapes
:���������PP�
2Encoder-SelfAttentionLayer-2/dropout/dropout/ShapeShape6Encoder-SelfAttentionLayer-2/softmax/Softmax:softmax:0*
T0*
_output_shapes
::���
IEncoder-SelfAttentionLayer-2/dropout/dropout/random_uniform/RandomUniformRandomUniform;Encoder-SelfAttentionLayer-2/dropout/dropout/Shape:output:0*
T0*/
_output_shapes
:���������PP*
dtype0*
seed���
;Encoder-SelfAttentionLayer-2/dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=�
9Encoder-SelfAttentionLayer-2/dropout/dropout/GreaterEqualGreaterEqualREncoder-SelfAttentionLayer-2/dropout/dropout/random_uniform/RandomUniform:output:0DEncoder-SelfAttentionLayer-2/dropout/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:���������PPy
4Encoder-SelfAttentionLayer-2/dropout/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
5Encoder-SelfAttentionLayer-2/dropout/dropout/SelectV2SelectV2=Encoder-SelfAttentionLayer-2/dropout/dropout/GreaterEqual:z:04Encoder-SelfAttentionLayer-2/dropout/dropout/Mul:z:0=Encoder-SelfAttentionLayer-2/dropout/dropout/Const_1:output:0*
T0*/
_output_shapes
:���������PP�
,Encoder-SelfAttentionLayer-2/einsum_1/EinsumEinsum>Encoder-SelfAttentionLayer-2/dropout/dropout/SelectV2:output:0*Encoder-SelfAttentionLayer-2/value/add:z:0*
N*
T0*/
_output_shapes
:���������P*
equationacbe,aecd->abcd�
JEncoder-SelfAttentionLayer-2/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpSencoder_selfattentionlayer_2_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:	*
dtype0�
;Encoder-SelfAttentionLayer-2/attention_output/einsum/EinsumEinsum5Encoder-SelfAttentionLayer-2/einsum_1/Einsum:output:0REncoder-SelfAttentionLayer-2/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:���������P	*
equationabcd,cde->abe�
@Encoder-SelfAttentionLayer-2/attention_output/add/ReadVariableOpReadVariableOpIencoder_selfattentionlayer_2_attention_output_add_readvariableop_resource*
_output_shapes
:	*
dtype0�
1Encoder-SelfAttentionLayer-2/attention_output/addAddV2DEncoder-SelfAttentionLayer-2/attention_output/einsum/Einsum:output:0HEncoder-SelfAttentionLayer-2/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������P	�
Encoder-1st-AdditionLayer-2/addAddV2inputs5Encoder-SelfAttentionLayer-2/attention_output/add:z:0*
T0*+
_output_shapes
:���������P	�
&Encoder-2nd-NormalizationLayer-2/ShapeShape#Encoder-1st-AdditionLayer-2/add:z:0*
T0*
_output_shapes
::��~
4Encoder-2nd-NormalizationLayer-2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
6Encoder-2nd-NormalizationLayer-2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
6Encoder-2nd-NormalizationLayer-2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
.Encoder-2nd-NormalizationLayer-2/strided_sliceStridedSlice/Encoder-2nd-NormalizationLayer-2/Shape:output:0=Encoder-2nd-NormalizationLayer-2/strided_slice/stack:output:0?Encoder-2nd-NormalizationLayer-2/strided_slice/stack_1:output:0?Encoder-2nd-NormalizationLayer-2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskp
&Encoder-2nd-NormalizationLayer-2/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
%Encoder-2nd-NormalizationLayer-2/ProdProd7Encoder-2nd-NormalizationLayer-2/strided_slice:output:0/Encoder-2nd-NormalizationLayer-2/Const:output:0*
T0*
_output_shapes
: �
6Encoder-2nd-NormalizationLayer-2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:�
8Encoder-2nd-NormalizationLayer-2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: �
8Encoder-2nd-NormalizationLayer-2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
0Encoder-2nd-NormalizationLayer-2/strided_slice_1StridedSlice/Encoder-2nd-NormalizationLayer-2/Shape:output:0?Encoder-2nd-NormalizationLayer-2/strided_slice_1/stack:output:0AEncoder-2nd-NormalizationLayer-2/strided_slice_1/stack_1:output:0AEncoder-2nd-NormalizationLayer-2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_maskr
(Encoder-2nd-NormalizationLayer-2/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
'Encoder-2nd-NormalizationLayer-2/Prod_1Prod9Encoder-2nd-NormalizationLayer-2/strided_slice_1:output:01Encoder-2nd-NormalizationLayer-2/Const_1:output:0*
T0*
_output_shapes
: r
0Encoder-2nd-NormalizationLayer-2/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :r
0Encoder-2nd-NormalizationLayer-2/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :�
.Encoder-2nd-NormalizationLayer-2/Reshape/shapePack9Encoder-2nd-NormalizationLayer-2/Reshape/shape/0:output:0.Encoder-2nd-NormalizationLayer-2/Prod:output:00Encoder-2nd-NormalizationLayer-2/Prod_1:output:09Encoder-2nd-NormalizationLayer-2/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:�
(Encoder-2nd-NormalizationLayer-2/ReshapeReshape#Encoder-1st-AdditionLayer-2/add:z:07Encoder-2nd-NormalizationLayer-2/Reshape/shape:output:0*
T0*8
_output_shapes&
$:"�������������������
,Encoder-2nd-NormalizationLayer-2/ones/packedPack.Encoder-2nd-NormalizationLayer-2/Prod:output:0*
N*
T0*
_output_shapes
:p
+Encoder-2nd-NormalizationLayer-2/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
%Encoder-2nd-NormalizationLayer-2/onesFill5Encoder-2nd-NormalizationLayer-2/ones/packed:output:04Encoder-2nd-NormalizationLayer-2/ones/Const:output:0*
T0*#
_output_shapes
:����������
-Encoder-2nd-NormalizationLayer-2/zeros/packedPack.Encoder-2nd-NormalizationLayer-2/Prod:output:0*
N*
T0*
_output_shapes
:q
,Encoder-2nd-NormalizationLayer-2/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
&Encoder-2nd-NormalizationLayer-2/zerosFill6Encoder-2nd-NormalizationLayer-2/zeros/packed:output:05Encoder-2nd-NormalizationLayer-2/zeros/Const:output:0*
T0*#
_output_shapes
:���������k
(Encoder-2nd-NormalizationLayer-2/Const_2Const*
_output_shapes
: *
dtype0*
valueB k
(Encoder-2nd-NormalizationLayer-2/Const_3Const*
_output_shapes
: *
dtype0*
valueB �
1Encoder-2nd-NormalizationLayer-2/FusedBatchNormV3FusedBatchNormV31Encoder-2nd-NormalizationLayer-2/Reshape:output:0.Encoder-2nd-NormalizationLayer-2/ones:output:0/Encoder-2nd-NormalizationLayer-2/zeros:output:01Encoder-2nd-NormalizationLayer-2/Const_2:output:01Encoder-2nd-NormalizationLayer-2/Const_3:output:0*
T0*
U0*x
_output_shapesf
d:"������������������:���������:���������:���������:���������:*
data_formatNCHW*
epsilon%o�:�
*Encoder-2nd-NormalizationLayer-2/Reshape_1Reshape5Encoder-2nd-NormalizationLayer-2/FusedBatchNormV3:y:0/Encoder-2nd-NormalizationLayer-2/Shape:output:0*
T0*+
_output_shapes
:���������P	�
3Encoder-2nd-NormalizationLayer-2/mul/ReadVariableOpReadVariableOp<encoder_2nd_normalizationlayer_2_mul_readvariableop_resource*
_output_shapes
:	*
dtype0�
$Encoder-2nd-NormalizationLayer-2/mulMul3Encoder-2nd-NormalizationLayer-2/Reshape_1:output:0;Encoder-2nd-NormalizationLayer-2/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������P	�
3Encoder-2nd-NormalizationLayer-2/add/ReadVariableOpReadVariableOp<encoder_2nd_normalizationlayer_2_add_readvariableop_resource*
_output_shapes
:	*
dtype0�
$Encoder-2nd-NormalizationLayer-2/addAddV2(Encoder-2nd-NormalizationLayer-2/mul:z:0;Encoder-2nd-NormalizationLayer-2/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������P	�
5Encoder-FeedForwardLayer_1_2/Tensordot/ReadVariableOpReadVariableOp>encoder_feedforwardlayer_1_2_tensordot_readvariableop_resource*
_output_shapes

:		*
dtype0u
+Encoder-FeedForwardLayer_1_2/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:|
+Encoder-FeedForwardLayer_1_2/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       �
,Encoder-FeedForwardLayer_1_2/Tensordot/ShapeShape(Encoder-2nd-NormalizationLayer-2/add:z:0*
T0*
_output_shapes
::��v
4Encoder-FeedForwardLayer_1_2/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
/Encoder-FeedForwardLayer_1_2/Tensordot/GatherV2GatherV25Encoder-FeedForwardLayer_1_2/Tensordot/Shape:output:04Encoder-FeedForwardLayer_1_2/Tensordot/free:output:0=Encoder-FeedForwardLayer_1_2/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:x
6Encoder-FeedForwardLayer_1_2/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
1Encoder-FeedForwardLayer_1_2/Tensordot/GatherV2_1GatherV25Encoder-FeedForwardLayer_1_2/Tensordot/Shape:output:04Encoder-FeedForwardLayer_1_2/Tensordot/axes:output:0?Encoder-FeedForwardLayer_1_2/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:v
,Encoder-FeedForwardLayer_1_2/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
+Encoder-FeedForwardLayer_1_2/Tensordot/ProdProd8Encoder-FeedForwardLayer_1_2/Tensordot/GatherV2:output:05Encoder-FeedForwardLayer_1_2/Tensordot/Const:output:0*
T0*
_output_shapes
: x
.Encoder-FeedForwardLayer_1_2/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
-Encoder-FeedForwardLayer_1_2/Tensordot/Prod_1Prod:Encoder-FeedForwardLayer_1_2/Tensordot/GatherV2_1:output:07Encoder-FeedForwardLayer_1_2/Tensordot/Const_1:output:0*
T0*
_output_shapes
: t
2Encoder-FeedForwardLayer_1_2/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
-Encoder-FeedForwardLayer_1_2/Tensordot/concatConcatV24Encoder-FeedForwardLayer_1_2/Tensordot/free:output:04Encoder-FeedForwardLayer_1_2/Tensordot/axes:output:0;Encoder-FeedForwardLayer_1_2/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
,Encoder-FeedForwardLayer_1_2/Tensordot/stackPack4Encoder-FeedForwardLayer_1_2/Tensordot/Prod:output:06Encoder-FeedForwardLayer_1_2/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
0Encoder-FeedForwardLayer_1_2/Tensordot/transpose	Transpose(Encoder-2nd-NormalizationLayer-2/add:z:06Encoder-FeedForwardLayer_1_2/Tensordot/concat:output:0*
T0*+
_output_shapes
:���������P	�
.Encoder-FeedForwardLayer_1_2/Tensordot/ReshapeReshape4Encoder-FeedForwardLayer_1_2/Tensordot/transpose:y:05Encoder-FeedForwardLayer_1_2/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
-Encoder-FeedForwardLayer_1_2/Tensordot/MatMulMatMul7Encoder-FeedForwardLayer_1_2/Tensordot/Reshape:output:0=Encoder-FeedForwardLayer_1_2/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������	x
.Encoder-FeedForwardLayer_1_2/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:	v
4Encoder-FeedForwardLayer_1_2/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
/Encoder-FeedForwardLayer_1_2/Tensordot/concat_1ConcatV28Encoder-FeedForwardLayer_1_2/Tensordot/GatherV2:output:07Encoder-FeedForwardLayer_1_2/Tensordot/Const_2:output:0=Encoder-FeedForwardLayer_1_2/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
&Encoder-FeedForwardLayer_1_2/TensordotReshape7Encoder-FeedForwardLayer_1_2/Tensordot/MatMul:product:08Encoder-FeedForwardLayer_1_2/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������P	�
3Encoder-FeedForwardLayer_1_2/BiasAdd/ReadVariableOpReadVariableOp<encoder_feedforwardlayer_1_2_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype0�
$Encoder-FeedForwardLayer_1_2/BiasAddBiasAdd/Encoder-FeedForwardLayer_1_2/Tensordot:output:0;Encoder-FeedForwardLayer_1_2/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������P	l
'Encoder-FeedForwardLayer_1_2/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
%Encoder-FeedForwardLayer_1_2/Gelu/mulMul0Encoder-FeedForwardLayer_1_2/Gelu/mul/x:output:0-Encoder-FeedForwardLayer_1_2/BiasAdd:output:0*
T0*+
_output_shapes
:���������P	m
(Encoder-FeedForwardLayer_1_2/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *��?�
)Encoder-FeedForwardLayer_1_2/Gelu/truedivRealDiv-Encoder-FeedForwardLayer_1_2/BiasAdd:output:01Encoder-FeedForwardLayer_1_2/Gelu/Cast/x:output:0*
T0*+
_output_shapes
:���������P	�
%Encoder-FeedForwardLayer_1_2/Gelu/ErfErf-Encoder-FeedForwardLayer_1_2/Gelu/truediv:z:0*
T0*+
_output_shapes
:���������P	l
'Encoder-FeedForwardLayer_1_2/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
%Encoder-FeedForwardLayer_1_2/Gelu/addAddV20Encoder-FeedForwardLayer_1_2/Gelu/add/x:output:0)Encoder-FeedForwardLayer_1_2/Gelu/Erf:y:0*
T0*+
_output_shapes
:���������P	�
'Encoder-FeedForwardLayer_1_2/Gelu/mul_1Mul)Encoder-FeedForwardLayer_1_2/Gelu/mul:z:0)Encoder-FeedForwardLayer_1_2/Gelu/add:z:0*
T0*+
_output_shapes
:���������P	�
5Encoder-FeedForwardLayer_2_2/Tensordot/ReadVariableOpReadVariableOp>encoder_feedforwardlayer_2_2_tensordot_readvariableop_resource*
_output_shapes

:		*
dtype0u
+Encoder-FeedForwardLayer_2_2/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:|
+Encoder-FeedForwardLayer_2_2/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       �
,Encoder-FeedForwardLayer_2_2/Tensordot/ShapeShape+Encoder-FeedForwardLayer_1_2/Gelu/mul_1:z:0*
T0*
_output_shapes
::��v
4Encoder-FeedForwardLayer_2_2/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
/Encoder-FeedForwardLayer_2_2/Tensordot/GatherV2GatherV25Encoder-FeedForwardLayer_2_2/Tensordot/Shape:output:04Encoder-FeedForwardLayer_2_2/Tensordot/free:output:0=Encoder-FeedForwardLayer_2_2/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:x
6Encoder-FeedForwardLayer_2_2/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
1Encoder-FeedForwardLayer_2_2/Tensordot/GatherV2_1GatherV25Encoder-FeedForwardLayer_2_2/Tensordot/Shape:output:04Encoder-FeedForwardLayer_2_2/Tensordot/axes:output:0?Encoder-FeedForwardLayer_2_2/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:v
,Encoder-FeedForwardLayer_2_2/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
+Encoder-FeedForwardLayer_2_2/Tensordot/ProdProd8Encoder-FeedForwardLayer_2_2/Tensordot/GatherV2:output:05Encoder-FeedForwardLayer_2_2/Tensordot/Const:output:0*
T0*
_output_shapes
: x
.Encoder-FeedForwardLayer_2_2/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
-Encoder-FeedForwardLayer_2_2/Tensordot/Prod_1Prod:Encoder-FeedForwardLayer_2_2/Tensordot/GatherV2_1:output:07Encoder-FeedForwardLayer_2_2/Tensordot/Const_1:output:0*
T0*
_output_shapes
: t
2Encoder-FeedForwardLayer_2_2/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
-Encoder-FeedForwardLayer_2_2/Tensordot/concatConcatV24Encoder-FeedForwardLayer_2_2/Tensordot/free:output:04Encoder-FeedForwardLayer_2_2/Tensordot/axes:output:0;Encoder-FeedForwardLayer_2_2/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
,Encoder-FeedForwardLayer_2_2/Tensordot/stackPack4Encoder-FeedForwardLayer_2_2/Tensordot/Prod:output:06Encoder-FeedForwardLayer_2_2/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
0Encoder-FeedForwardLayer_2_2/Tensordot/transpose	Transpose+Encoder-FeedForwardLayer_1_2/Gelu/mul_1:z:06Encoder-FeedForwardLayer_2_2/Tensordot/concat:output:0*
T0*+
_output_shapes
:���������P	�
.Encoder-FeedForwardLayer_2_2/Tensordot/ReshapeReshape4Encoder-FeedForwardLayer_2_2/Tensordot/transpose:y:05Encoder-FeedForwardLayer_2_2/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
-Encoder-FeedForwardLayer_2_2/Tensordot/MatMulMatMul7Encoder-FeedForwardLayer_2_2/Tensordot/Reshape:output:0=Encoder-FeedForwardLayer_2_2/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������	x
.Encoder-FeedForwardLayer_2_2/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:	v
4Encoder-FeedForwardLayer_2_2/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
/Encoder-FeedForwardLayer_2_2/Tensordot/concat_1ConcatV28Encoder-FeedForwardLayer_2_2/Tensordot/GatherV2:output:07Encoder-FeedForwardLayer_2_2/Tensordot/Const_2:output:0=Encoder-FeedForwardLayer_2_2/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
&Encoder-FeedForwardLayer_2_2/TensordotReshape7Encoder-FeedForwardLayer_2_2/Tensordot/MatMul:product:08Encoder-FeedForwardLayer_2_2/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������P	�
3Encoder-FeedForwardLayer_2_2/BiasAdd/ReadVariableOpReadVariableOp<encoder_feedforwardlayer_2_2_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype0�
$Encoder-FeedForwardLayer_2_2/BiasAddBiasAdd/Encoder-FeedForwardLayer_2_2/Tensordot:output:0;Encoder-FeedForwardLayer_2_2/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������P	\
dropout_4/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?�
dropout_4/dropout/MulMul-Encoder-FeedForwardLayer_2_2/BiasAdd:output:0 dropout_4/dropout/Const:output:0*
T0*+
_output_shapes
:���������P	�
dropout_4/dropout/ShapeShape-Encoder-FeedForwardLayer_2_2/BiasAdd:output:0*
T0*
_output_shapes
::���
.dropout_4/dropout/random_uniform/RandomUniformRandomUniform dropout_4/dropout/Shape:output:0*
T0*+
_output_shapes
:���������P	*
dtype0*
seed2*
seed��e
 dropout_4/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=�
dropout_4/dropout/GreaterEqualGreaterEqual7dropout_4/dropout/random_uniform/RandomUniform:output:0)dropout_4/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:���������P	^
dropout_4/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout_4/dropout/SelectV2SelectV2"dropout_4/dropout/GreaterEqual:z:0dropout_4/dropout/Mul:z:0"dropout_4/dropout/Const_1:output:0*
T0*+
_output_shapes
:���������P	�
Encoder-2nd-AdditionLayer-2/addAddV2#Encoder-1st-AdditionLayer-2/add:z:0#dropout_4/dropout/SelectV2:output:0*
T0*+
_output_shapes
:���������P	v
IdentityIdentity#Encoder-2nd-AdditionLayer-2/add:z:0^NoOp*
T0*+
_output_shapes
:���������P	�

Identity_1Identity6Encoder-SelfAttentionLayer-2/softmax/Softmax:softmax:0^NoOp*
T0*/
_output_shapes
:���������PP�
NoOpNoOp4^Encoder-1st-NormalizationLayer-2/add/ReadVariableOp4^Encoder-1st-NormalizationLayer-2/mul/ReadVariableOp4^Encoder-2nd-NormalizationLayer-2/add/ReadVariableOp4^Encoder-2nd-NormalizationLayer-2/mul/ReadVariableOp4^Encoder-FeedForwardLayer_1_2/BiasAdd/ReadVariableOp6^Encoder-FeedForwardLayer_1_2/Tensordot/ReadVariableOp4^Encoder-FeedForwardLayer_2_2/BiasAdd/ReadVariableOp6^Encoder-FeedForwardLayer_2_2/Tensordot/ReadVariableOpA^Encoder-SelfAttentionLayer-2/attention_output/add/ReadVariableOpK^Encoder-SelfAttentionLayer-2/attention_output/einsum/Einsum/ReadVariableOp4^Encoder-SelfAttentionLayer-2/key/add/ReadVariableOp>^Encoder-SelfAttentionLayer-2/key/einsum/Einsum/ReadVariableOp6^Encoder-SelfAttentionLayer-2/query/add/ReadVariableOp@^Encoder-SelfAttentionLayer-2/query/einsum/Einsum/ReadVariableOp6^Encoder-SelfAttentionLayer-2/value/add/ReadVariableOp@^Encoder-SelfAttentionLayer-2/value/einsum/Einsum/ReadVariableOp*
_output_shapes
 "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:���������P	: : : : : : : : : : : : : : : : 2j
3Encoder-1st-NormalizationLayer-2/add/ReadVariableOp3Encoder-1st-NormalizationLayer-2/add/ReadVariableOp2j
3Encoder-1st-NormalizationLayer-2/mul/ReadVariableOp3Encoder-1st-NormalizationLayer-2/mul/ReadVariableOp2j
3Encoder-2nd-NormalizationLayer-2/add/ReadVariableOp3Encoder-2nd-NormalizationLayer-2/add/ReadVariableOp2j
3Encoder-2nd-NormalizationLayer-2/mul/ReadVariableOp3Encoder-2nd-NormalizationLayer-2/mul/ReadVariableOp2j
3Encoder-FeedForwardLayer_1_2/BiasAdd/ReadVariableOp3Encoder-FeedForwardLayer_1_2/BiasAdd/ReadVariableOp2n
5Encoder-FeedForwardLayer_1_2/Tensordot/ReadVariableOp5Encoder-FeedForwardLayer_1_2/Tensordot/ReadVariableOp2j
3Encoder-FeedForwardLayer_2_2/BiasAdd/ReadVariableOp3Encoder-FeedForwardLayer_2_2/BiasAdd/ReadVariableOp2n
5Encoder-FeedForwardLayer_2_2/Tensordot/ReadVariableOp5Encoder-FeedForwardLayer_2_2/Tensordot/ReadVariableOp2�
@Encoder-SelfAttentionLayer-2/attention_output/add/ReadVariableOp@Encoder-SelfAttentionLayer-2/attention_output/add/ReadVariableOp2�
JEncoder-SelfAttentionLayer-2/attention_output/einsum/Einsum/ReadVariableOpJEncoder-SelfAttentionLayer-2/attention_output/einsum/Einsum/ReadVariableOp2j
3Encoder-SelfAttentionLayer-2/key/add/ReadVariableOp3Encoder-SelfAttentionLayer-2/key/add/ReadVariableOp2~
=Encoder-SelfAttentionLayer-2/key/einsum/Einsum/ReadVariableOp=Encoder-SelfAttentionLayer-2/key/einsum/Einsum/ReadVariableOp2n
5Encoder-SelfAttentionLayer-2/query/add/ReadVariableOp5Encoder-SelfAttentionLayer-2/query/add/ReadVariableOp2�
?Encoder-SelfAttentionLayer-2/query/einsum/Einsum/ReadVariableOp?Encoder-SelfAttentionLayer-2/query/einsum/Einsum/ReadVariableOp2n
5Encoder-SelfAttentionLayer-2/value/add/ReadVariableOp5Encoder-SelfAttentionLayer-2/value/add/ReadVariableOp2�
?Encoder-SelfAttentionLayer-2/value/einsum/Einsum/ReadVariableOp?Encoder-SelfAttentionLayer-2/value/einsum/Einsum/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:(
$
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:S O
+
_output_shapes
:���������P	
 
_user_specified_nameinputs
�
^
B__inference_Output_layer_call_and_return_conditional_losses_233387

inputs
identity=
SqueezeSqueezeinputs*
T0*
_output_shapes
:I
IdentityIdentitySqueeze:output:0*
T0*
_output_shapes
:"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�	
�S
!__inference__wrapped_model_229753
stacklevelinputfeatures
timelimitinputh
Zmodel_1_transformer_encoder_3_encoder_1st_normalizationlayer_1_mul_readvariableop_resource:	h
Zmodel_1_transformer_encoder_3_encoder_1st_normalizationlayer_1_add_readvariableop_resource:	|
fmodel_1_transformer_encoder_3_encoder_selfattentionlayer_1_query_einsum_einsum_readvariableop_resource:	n
\model_1_transformer_encoder_3_encoder_selfattentionlayer_1_query_add_readvariableop_resource:z
dmodel_1_transformer_encoder_3_encoder_selfattentionlayer_1_key_einsum_einsum_readvariableop_resource:	l
Zmodel_1_transformer_encoder_3_encoder_selfattentionlayer_1_key_add_readvariableop_resource:|
fmodel_1_transformer_encoder_3_encoder_selfattentionlayer_1_value_einsum_einsum_readvariableop_resource:	n
\model_1_transformer_encoder_3_encoder_selfattentionlayer_1_value_add_readvariableop_resource:�
qmodel_1_transformer_encoder_3_encoder_selfattentionlayer_1_attention_output_einsum_einsum_readvariableop_resource:	u
gmodel_1_transformer_encoder_3_encoder_selfattentionlayer_1_attention_output_add_readvariableop_resource:	h
Zmodel_1_transformer_encoder_3_encoder_2nd_normalizationlayer_1_mul_readvariableop_resource:	h
Zmodel_1_transformer_encoder_3_encoder_2nd_normalizationlayer_1_add_readvariableop_resource:	n
\model_1_transformer_encoder_3_encoder_feedforwardlayer_1_1_tensordot_readvariableop_resource:		h
Zmodel_1_transformer_encoder_3_encoder_feedforwardlayer_1_1_biasadd_readvariableop_resource:	n
\model_1_transformer_encoder_3_encoder_feedforwardlayer_2_1_tensordot_readvariableop_resource:		h
Zmodel_1_transformer_encoder_3_encoder_feedforwardlayer_2_1_biasadd_readvariableop_resource:	h
Zmodel_1_transformer_encoder_4_encoder_1st_normalizationlayer_2_mul_readvariableop_resource:	h
Zmodel_1_transformer_encoder_4_encoder_1st_normalizationlayer_2_add_readvariableop_resource:	|
fmodel_1_transformer_encoder_4_encoder_selfattentionlayer_2_query_einsum_einsum_readvariableop_resource:	n
\model_1_transformer_encoder_4_encoder_selfattentionlayer_2_query_add_readvariableop_resource:z
dmodel_1_transformer_encoder_4_encoder_selfattentionlayer_2_key_einsum_einsum_readvariableop_resource:	l
Zmodel_1_transformer_encoder_4_encoder_selfattentionlayer_2_key_add_readvariableop_resource:|
fmodel_1_transformer_encoder_4_encoder_selfattentionlayer_2_value_einsum_einsum_readvariableop_resource:	n
\model_1_transformer_encoder_4_encoder_selfattentionlayer_2_value_add_readvariableop_resource:�
qmodel_1_transformer_encoder_4_encoder_selfattentionlayer_2_attention_output_einsum_einsum_readvariableop_resource:	u
gmodel_1_transformer_encoder_4_encoder_selfattentionlayer_2_attention_output_add_readvariableop_resource:	h
Zmodel_1_transformer_encoder_4_encoder_2nd_normalizationlayer_2_mul_readvariableop_resource:	h
Zmodel_1_transformer_encoder_4_encoder_2nd_normalizationlayer_2_add_readvariableop_resource:	n
\model_1_transformer_encoder_4_encoder_feedforwardlayer_1_2_tensordot_readvariableop_resource:		h
Zmodel_1_transformer_encoder_4_encoder_feedforwardlayer_1_2_biasadd_readvariableop_resource:	n
\model_1_transformer_encoder_4_encoder_feedforwardlayer_2_2_tensordot_readvariableop_resource:		h
Zmodel_1_transformer_encoder_4_encoder_feedforwardlayer_2_2_biasadd_readvariableop_resource:	h
Zmodel_1_transformer_encoder_5_encoder_1st_normalizationlayer_3_mul_readvariableop_resource:	h
Zmodel_1_transformer_encoder_5_encoder_1st_normalizationlayer_3_add_readvariableop_resource:	|
fmodel_1_transformer_encoder_5_encoder_selfattentionlayer_3_query_einsum_einsum_readvariableop_resource:	n
\model_1_transformer_encoder_5_encoder_selfattentionlayer_3_query_add_readvariableop_resource:z
dmodel_1_transformer_encoder_5_encoder_selfattentionlayer_3_key_einsum_einsum_readvariableop_resource:	l
Zmodel_1_transformer_encoder_5_encoder_selfattentionlayer_3_key_add_readvariableop_resource:|
fmodel_1_transformer_encoder_5_encoder_selfattentionlayer_3_value_einsum_einsum_readvariableop_resource:	n
\model_1_transformer_encoder_5_encoder_selfattentionlayer_3_value_add_readvariableop_resource:�
qmodel_1_transformer_encoder_5_encoder_selfattentionlayer_3_attention_output_einsum_einsum_readvariableop_resource:	u
gmodel_1_transformer_encoder_5_encoder_selfattentionlayer_3_attention_output_add_readvariableop_resource:	h
Zmodel_1_transformer_encoder_5_encoder_2nd_normalizationlayer_3_mul_readvariableop_resource:	h
Zmodel_1_transformer_encoder_5_encoder_2nd_normalizationlayer_3_add_readvariableop_resource:	n
\model_1_transformer_encoder_5_encoder_feedforwardlayer_1_3_tensordot_readvariableop_resource:		h
Zmodel_1_transformer_encoder_5_encoder_feedforwardlayer_1_3_biasadd_readvariableop_resource:	n
\model_1_transformer_encoder_5_encoder_feedforwardlayer_2_3_tensordot_readvariableop_resource:		h
Zmodel_1_transformer_encoder_5_encoder_feedforwardlayer_2_3_biasadd_readvariableop_resource:	@
2model_1_finallayernorm_mul_readvariableop_resource:	@
2model_1_finallayernorm_add_readvariableop_resource:	W
Emodel_1_fullyconnectedlayerimprovement_matmul_readvariableop_resource:

T
Fmodel_1_fullyconnectedlayerimprovement_biasadd_readvariableop_resource:
N
<model_1_predictionimprovement_matmul_readvariableop_resource:
K
=model_1_predictionimprovement_biasadd_readvariableop_resource:
identity��)model_1/FinalLayerNorm/add/ReadVariableOp�)model_1/FinalLayerNorm/mul/ReadVariableOp�=model_1/FullyConnectedLayerImprovement/BiasAdd/ReadVariableOp�<model_1/FullyConnectedLayerImprovement/MatMul/ReadVariableOp�4model_1/PredictionImprovement/BiasAdd/ReadVariableOp�3model_1/PredictionImprovement/MatMul/ReadVariableOp�Qmodel_1/transformer_encoder_3/Encoder-1st-NormalizationLayer-1/add/ReadVariableOp�Qmodel_1/transformer_encoder_3/Encoder-1st-NormalizationLayer-1/mul/ReadVariableOp�Qmodel_1/transformer_encoder_3/Encoder-2nd-NormalizationLayer-1/add/ReadVariableOp�Qmodel_1/transformer_encoder_3/Encoder-2nd-NormalizationLayer-1/mul/ReadVariableOp�Qmodel_1/transformer_encoder_3/Encoder-FeedForwardLayer_1_1/BiasAdd/ReadVariableOp�Smodel_1/transformer_encoder_3/Encoder-FeedForwardLayer_1_1/Tensordot/ReadVariableOp�Qmodel_1/transformer_encoder_3/Encoder-FeedForwardLayer_2_1/BiasAdd/ReadVariableOp�Smodel_1/transformer_encoder_3/Encoder-FeedForwardLayer_2_1/Tensordot/ReadVariableOp�^model_1/transformer_encoder_3/Encoder-SelfAttentionLayer-1/attention_output/add/ReadVariableOp�hmodel_1/transformer_encoder_3/Encoder-SelfAttentionLayer-1/attention_output/einsum/Einsum/ReadVariableOp�Qmodel_1/transformer_encoder_3/Encoder-SelfAttentionLayer-1/key/add/ReadVariableOp�[model_1/transformer_encoder_3/Encoder-SelfAttentionLayer-1/key/einsum/Einsum/ReadVariableOp�Smodel_1/transformer_encoder_3/Encoder-SelfAttentionLayer-1/query/add/ReadVariableOp�]model_1/transformer_encoder_3/Encoder-SelfAttentionLayer-1/query/einsum/Einsum/ReadVariableOp�Smodel_1/transformer_encoder_3/Encoder-SelfAttentionLayer-1/value/add/ReadVariableOp�]model_1/transformer_encoder_3/Encoder-SelfAttentionLayer-1/value/einsum/Einsum/ReadVariableOp�Qmodel_1/transformer_encoder_4/Encoder-1st-NormalizationLayer-2/add/ReadVariableOp�Qmodel_1/transformer_encoder_4/Encoder-1st-NormalizationLayer-2/mul/ReadVariableOp�Qmodel_1/transformer_encoder_4/Encoder-2nd-NormalizationLayer-2/add/ReadVariableOp�Qmodel_1/transformer_encoder_4/Encoder-2nd-NormalizationLayer-2/mul/ReadVariableOp�Qmodel_1/transformer_encoder_4/Encoder-FeedForwardLayer_1_2/BiasAdd/ReadVariableOp�Smodel_1/transformer_encoder_4/Encoder-FeedForwardLayer_1_2/Tensordot/ReadVariableOp�Qmodel_1/transformer_encoder_4/Encoder-FeedForwardLayer_2_2/BiasAdd/ReadVariableOp�Smodel_1/transformer_encoder_4/Encoder-FeedForwardLayer_2_2/Tensordot/ReadVariableOp�^model_1/transformer_encoder_4/Encoder-SelfAttentionLayer-2/attention_output/add/ReadVariableOp�hmodel_1/transformer_encoder_4/Encoder-SelfAttentionLayer-2/attention_output/einsum/Einsum/ReadVariableOp�Qmodel_1/transformer_encoder_4/Encoder-SelfAttentionLayer-2/key/add/ReadVariableOp�[model_1/transformer_encoder_4/Encoder-SelfAttentionLayer-2/key/einsum/Einsum/ReadVariableOp�Smodel_1/transformer_encoder_4/Encoder-SelfAttentionLayer-2/query/add/ReadVariableOp�]model_1/transformer_encoder_4/Encoder-SelfAttentionLayer-2/query/einsum/Einsum/ReadVariableOp�Smodel_1/transformer_encoder_4/Encoder-SelfAttentionLayer-2/value/add/ReadVariableOp�]model_1/transformer_encoder_4/Encoder-SelfAttentionLayer-2/value/einsum/Einsum/ReadVariableOp�Qmodel_1/transformer_encoder_5/Encoder-1st-NormalizationLayer-3/add/ReadVariableOp�Qmodel_1/transformer_encoder_5/Encoder-1st-NormalizationLayer-3/mul/ReadVariableOp�Qmodel_1/transformer_encoder_5/Encoder-2nd-NormalizationLayer-3/add/ReadVariableOp�Qmodel_1/transformer_encoder_5/Encoder-2nd-NormalizationLayer-3/mul/ReadVariableOp�Qmodel_1/transformer_encoder_5/Encoder-FeedForwardLayer_1_3/BiasAdd/ReadVariableOp�Smodel_1/transformer_encoder_5/Encoder-FeedForwardLayer_1_3/Tensordot/ReadVariableOp�Qmodel_1/transformer_encoder_5/Encoder-FeedForwardLayer_2_3/BiasAdd/ReadVariableOp�Smodel_1/transformer_encoder_5/Encoder-FeedForwardLayer_2_3/Tensordot/ReadVariableOp�^model_1/transformer_encoder_5/Encoder-SelfAttentionLayer-3/attention_output/add/ReadVariableOp�hmodel_1/transformer_encoder_5/Encoder-SelfAttentionLayer-3/attention_output/einsum/Einsum/ReadVariableOp�Qmodel_1/transformer_encoder_5/Encoder-SelfAttentionLayer-3/key/add/ReadVariableOp�[model_1/transformer_encoder_5/Encoder-SelfAttentionLayer-3/key/einsum/Einsum/ReadVariableOp�Smodel_1/transformer_encoder_5/Encoder-SelfAttentionLayer-3/query/add/ReadVariableOp�]model_1/transformer_encoder_5/Encoder-SelfAttentionLayer-3/query/einsum/Einsum/ReadVariableOp�Smodel_1/transformer_encoder_5/Encoder-SelfAttentionLayer-3/value/add/ReadVariableOp�]model_1/transformer_encoder_5/Encoder-SelfAttentionLayer-3/value/einsum/Einsum/ReadVariableOpa
model_1/MaskingLayer/NotEqual/yConst*
_output_shapes
: *
dtype0*
value	B j �
model_1/MaskingLayer/NotEqualNotEqualstacklevelinputfeatures(model_1/MaskingLayer/NotEqual/y:output:0*
T0*+
_output_shapes
:���������P	u
*model_1/MaskingLayer/Any/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
����������
model_1/MaskingLayer/AnyAny!model_1/MaskingLayer/NotEqual:z:03model_1/MaskingLayer/Any/reduction_indices:output:0*+
_output_shapes
:���������P*
	keep_dims(�
model_1/MaskingLayer/CastCast!model_1/MaskingLayer/Any:output:0*

DstT0*

SrcT0
*+
_output_shapes
:���������P�
model_1/MaskingLayer/mulMulstacklevelinputfeaturesmodel_1/MaskingLayer/Cast:y:0*
T0*+
_output_shapes
:���������P	�
model_1/MaskingLayer/SqueezeSqueeze!model_1/MaskingLayer/Any:output:0*
T0
*'
_output_shapes
:���������P*
squeeze_dims

����������
"model_1/transformer_encoder_3/CastCastmodel_1/MaskingLayer/mul:z:0*

DstT0*

SrcT0*+
_output_shapes
:���������P	�
Dmodel_1/transformer_encoder_3/Encoder-1st-NormalizationLayer-1/ShapeShape&model_1/transformer_encoder_3/Cast:y:0*
T0*
_output_shapes
::���
Rmodel_1/transformer_encoder_3/Encoder-1st-NormalizationLayer-1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
Tmodel_1/transformer_encoder_3/Encoder-1st-NormalizationLayer-1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
Tmodel_1/transformer_encoder_3/Encoder-1st-NormalizationLayer-1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
Lmodel_1/transformer_encoder_3/Encoder-1st-NormalizationLayer-1/strided_sliceStridedSliceMmodel_1/transformer_encoder_3/Encoder-1st-NormalizationLayer-1/Shape:output:0[model_1/transformer_encoder_3/Encoder-1st-NormalizationLayer-1/strided_slice/stack:output:0]model_1/transformer_encoder_3/Encoder-1st-NormalizationLayer-1/strided_slice/stack_1:output:0]model_1/transformer_encoder_3/Encoder-1st-NormalizationLayer-1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask�
Dmodel_1/transformer_encoder_3/Encoder-1st-NormalizationLayer-1/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
Cmodel_1/transformer_encoder_3/Encoder-1st-NormalizationLayer-1/ProdProdUmodel_1/transformer_encoder_3/Encoder-1st-NormalizationLayer-1/strided_slice:output:0Mmodel_1/transformer_encoder_3/Encoder-1st-NormalizationLayer-1/Const:output:0*
T0*
_output_shapes
: �
Tmodel_1/transformer_encoder_3/Encoder-1st-NormalizationLayer-1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:�
Vmodel_1/transformer_encoder_3/Encoder-1st-NormalizationLayer-1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: �
Vmodel_1/transformer_encoder_3/Encoder-1st-NormalizationLayer-1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
Nmodel_1/transformer_encoder_3/Encoder-1st-NormalizationLayer-1/strided_slice_1StridedSliceMmodel_1/transformer_encoder_3/Encoder-1st-NormalizationLayer-1/Shape:output:0]model_1/transformer_encoder_3/Encoder-1st-NormalizationLayer-1/strided_slice_1/stack:output:0_model_1/transformer_encoder_3/Encoder-1st-NormalizationLayer-1/strided_slice_1/stack_1:output:0_model_1/transformer_encoder_3/Encoder-1st-NormalizationLayer-1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask�
Fmodel_1/transformer_encoder_3/Encoder-1st-NormalizationLayer-1/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
Emodel_1/transformer_encoder_3/Encoder-1st-NormalizationLayer-1/Prod_1ProdWmodel_1/transformer_encoder_3/Encoder-1st-NormalizationLayer-1/strided_slice_1:output:0Omodel_1/transformer_encoder_3/Encoder-1st-NormalizationLayer-1/Const_1:output:0*
T0*
_output_shapes
: �
Nmodel_1/transformer_encoder_3/Encoder-1st-NormalizationLayer-1/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :�
Nmodel_1/transformer_encoder_3/Encoder-1st-NormalizationLayer-1/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :�
Lmodel_1/transformer_encoder_3/Encoder-1st-NormalizationLayer-1/Reshape/shapePackWmodel_1/transformer_encoder_3/Encoder-1st-NormalizationLayer-1/Reshape/shape/0:output:0Lmodel_1/transformer_encoder_3/Encoder-1st-NormalizationLayer-1/Prod:output:0Nmodel_1/transformer_encoder_3/Encoder-1st-NormalizationLayer-1/Prod_1:output:0Wmodel_1/transformer_encoder_3/Encoder-1st-NormalizationLayer-1/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:�
Fmodel_1/transformer_encoder_3/Encoder-1st-NormalizationLayer-1/ReshapeReshape&model_1/transformer_encoder_3/Cast:y:0Umodel_1/transformer_encoder_3/Encoder-1st-NormalizationLayer-1/Reshape/shape:output:0*
T0*8
_output_shapes&
$:"�������������������
Jmodel_1/transformer_encoder_3/Encoder-1st-NormalizationLayer-1/ones/packedPackLmodel_1/transformer_encoder_3/Encoder-1st-NormalizationLayer-1/Prod:output:0*
N*
T0*
_output_shapes
:�
Imodel_1/transformer_encoder_3/Encoder-1st-NormalizationLayer-1/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
Cmodel_1/transformer_encoder_3/Encoder-1st-NormalizationLayer-1/onesFillSmodel_1/transformer_encoder_3/Encoder-1st-NormalizationLayer-1/ones/packed:output:0Rmodel_1/transformer_encoder_3/Encoder-1st-NormalizationLayer-1/ones/Const:output:0*
T0*#
_output_shapes
:����������
Kmodel_1/transformer_encoder_3/Encoder-1st-NormalizationLayer-1/zeros/packedPackLmodel_1/transformer_encoder_3/Encoder-1st-NormalizationLayer-1/Prod:output:0*
N*
T0*
_output_shapes
:�
Jmodel_1/transformer_encoder_3/Encoder-1st-NormalizationLayer-1/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
Dmodel_1/transformer_encoder_3/Encoder-1st-NormalizationLayer-1/zerosFillTmodel_1/transformer_encoder_3/Encoder-1st-NormalizationLayer-1/zeros/packed:output:0Smodel_1/transformer_encoder_3/Encoder-1st-NormalizationLayer-1/zeros/Const:output:0*
T0*#
_output_shapes
:����������
Fmodel_1/transformer_encoder_3/Encoder-1st-NormalizationLayer-1/Const_2Const*
_output_shapes
: *
dtype0*
valueB �
Fmodel_1/transformer_encoder_3/Encoder-1st-NormalizationLayer-1/Const_3Const*
_output_shapes
: *
dtype0*
valueB �
Omodel_1/transformer_encoder_3/Encoder-1st-NormalizationLayer-1/FusedBatchNormV3FusedBatchNormV3Omodel_1/transformer_encoder_3/Encoder-1st-NormalizationLayer-1/Reshape:output:0Lmodel_1/transformer_encoder_3/Encoder-1st-NormalizationLayer-1/ones:output:0Mmodel_1/transformer_encoder_3/Encoder-1st-NormalizationLayer-1/zeros:output:0Omodel_1/transformer_encoder_3/Encoder-1st-NormalizationLayer-1/Const_2:output:0Omodel_1/transformer_encoder_3/Encoder-1st-NormalizationLayer-1/Const_3:output:0*
T0*
U0*x
_output_shapesf
d:"������������������:���������:���������:���������:���������:*
data_formatNCHW*
epsilon%o�:�
Hmodel_1/transformer_encoder_3/Encoder-1st-NormalizationLayer-1/Reshape_1ReshapeSmodel_1/transformer_encoder_3/Encoder-1st-NormalizationLayer-1/FusedBatchNormV3:y:0Mmodel_1/transformer_encoder_3/Encoder-1st-NormalizationLayer-1/Shape:output:0*
T0*+
_output_shapes
:���������P	�
Qmodel_1/transformer_encoder_3/Encoder-1st-NormalizationLayer-1/mul/ReadVariableOpReadVariableOpZmodel_1_transformer_encoder_3_encoder_1st_normalizationlayer_1_mul_readvariableop_resource*
_output_shapes
:	*
dtype0�
Bmodel_1/transformer_encoder_3/Encoder-1st-NormalizationLayer-1/mulMulQmodel_1/transformer_encoder_3/Encoder-1st-NormalizationLayer-1/Reshape_1:output:0Ymodel_1/transformer_encoder_3/Encoder-1st-NormalizationLayer-1/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������P	�
Qmodel_1/transformer_encoder_3/Encoder-1st-NormalizationLayer-1/add/ReadVariableOpReadVariableOpZmodel_1_transformer_encoder_3_encoder_1st_normalizationlayer_1_add_readvariableop_resource*
_output_shapes
:	*
dtype0�
Bmodel_1/transformer_encoder_3/Encoder-1st-NormalizationLayer-1/addAddV2Fmodel_1/transformer_encoder_3/Encoder-1st-NormalizationLayer-1/mul:z:0Ymodel_1/transformer_encoder_3/Encoder-1st-NormalizationLayer-1/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������P	�
]model_1/transformer_encoder_3/Encoder-SelfAttentionLayer-1/query/einsum/Einsum/ReadVariableOpReadVariableOpfmodel_1_transformer_encoder_3_encoder_selfattentionlayer_1_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:	*
dtype0�
Nmodel_1/transformer_encoder_3/Encoder-SelfAttentionLayer-1/query/einsum/EinsumEinsumFmodel_1/transformer_encoder_3/Encoder-1st-NormalizationLayer-1/add:z:0emodel_1/transformer_encoder_3/Encoder-SelfAttentionLayer-1/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������P*
equationabc,cde->abde�
Smodel_1/transformer_encoder_3/Encoder-SelfAttentionLayer-1/query/add/ReadVariableOpReadVariableOp\model_1_transformer_encoder_3_encoder_selfattentionlayer_1_query_add_readvariableop_resource*
_output_shapes

:*
dtype0�
Dmodel_1/transformer_encoder_3/Encoder-SelfAttentionLayer-1/query/addAddV2Wmodel_1/transformer_encoder_3/Encoder-SelfAttentionLayer-1/query/einsum/Einsum:output:0[model_1/transformer_encoder_3/Encoder-SelfAttentionLayer-1/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������P�
[model_1/transformer_encoder_3/Encoder-SelfAttentionLayer-1/key/einsum/Einsum/ReadVariableOpReadVariableOpdmodel_1_transformer_encoder_3_encoder_selfattentionlayer_1_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:	*
dtype0�
Lmodel_1/transformer_encoder_3/Encoder-SelfAttentionLayer-1/key/einsum/EinsumEinsumFmodel_1/transformer_encoder_3/Encoder-1st-NormalizationLayer-1/add:z:0cmodel_1/transformer_encoder_3/Encoder-SelfAttentionLayer-1/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������P*
equationabc,cde->abde�
Qmodel_1/transformer_encoder_3/Encoder-SelfAttentionLayer-1/key/add/ReadVariableOpReadVariableOpZmodel_1_transformer_encoder_3_encoder_selfattentionlayer_1_key_add_readvariableop_resource*
_output_shapes

:*
dtype0�
Bmodel_1/transformer_encoder_3/Encoder-SelfAttentionLayer-1/key/addAddV2Umodel_1/transformer_encoder_3/Encoder-SelfAttentionLayer-1/key/einsum/Einsum:output:0Ymodel_1/transformer_encoder_3/Encoder-SelfAttentionLayer-1/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������P�
]model_1/transformer_encoder_3/Encoder-SelfAttentionLayer-1/value/einsum/Einsum/ReadVariableOpReadVariableOpfmodel_1_transformer_encoder_3_encoder_selfattentionlayer_1_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:	*
dtype0�
Nmodel_1/transformer_encoder_3/Encoder-SelfAttentionLayer-1/value/einsum/EinsumEinsumFmodel_1/transformer_encoder_3/Encoder-1st-NormalizationLayer-1/add:z:0emodel_1/transformer_encoder_3/Encoder-SelfAttentionLayer-1/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������P*
equationabc,cde->abde�
Smodel_1/transformer_encoder_3/Encoder-SelfAttentionLayer-1/value/add/ReadVariableOpReadVariableOp\model_1_transformer_encoder_3_encoder_selfattentionlayer_1_value_add_readvariableop_resource*
_output_shapes

:*
dtype0�
Dmodel_1/transformer_encoder_3/Encoder-SelfAttentionLayer-1/value/addAddV2Wmodel_1/transformer_encoder_3/Encoder-SelfAttentionLayer-1/value/einsum/Einsum:output:0[model_1/transformer_encoder_3/Encoder-SelfAttentionLayer-1/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������P�
@model_1/transformer_encoder_3/Encoder-SelfAttentionLayer-1/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *:�?�
>model_1/transformer_encoder_3/Encoder-SelfAttentionLayer-1/MulMulHmodel_1/transformer_encoder_3/Encoder-SelfAttentionLayer-1/query/add:z:0Imodel_1/transformer_encoder_3/Encoder-SelfAttentionLayer-1/Mul/y:output:0*
T0*/
_output_shapes
:���������P�
Hmodel_1/transformer_encoder_3/Encoder-SelfAttentionLayer-1/einsum/EinsumEinsumFmodel_1/transformer_encoder_3/Encoder-SelfAttentionLayer-1/key/add:z:0Bmodel_1/transformer_encoder_3/Encoder-SelfAttentionLayer-1/Mul:z:0*
N*
T0*/
_output_shapes
:���������PP*
equationaecd,abcd->acbe�
Jmodel_1/transformer_encoder_3/Encoder-SelfAttentionLayer-1/softmax/SoftmaxSoftmaxQmodel_1/transformer_encoder_3/Encoder-SelfAttentionLayer-1/einsum/Einsum:output:0*
T0*/
_output_shapes
:���������PP�
Kmodel_1/transformer_encoder_3/Encoder-SelfAttentionLayer-1/dropout/IdentityIdentityTmodel_1/transformer_encoder_3/Encoder-SelfAttentionLayer-1/softmax/Softmax:softmax:0*
T0*/
_output_shapes
:���������PP�
Jmodel_1/transformer_encoder_3/Encoder-SelfAttentionLayer-1/einsum_1/EinsumEinsumTmodel_1/transformer_encoder_3/Encoder-SelfAttentionLayer-1/dropout/Identity:output:0Hmodel_1/transformer_encoder_3/Encoder-SelfAttentionLayer-1/value/add:z:0*
N*
T0*/
_output_shapes
:���������P*
equationacbe,aecd->abcd�
hmodel_1/transformer_encoder_3/Encoder-SelfAttentionLayer-1/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpqmodel_1_transformer_encoder_3_encoder_selfattentionlayer_1_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:	*
dtype0�
Ymodel_1/transformer_encoder_3/Encoder-SelfAttentionLayer-1/attention_output/einsum/EinsumEinsumSmodel_1/transformer_encoder_3/Encoder-SelfAttentionLayer-1/einsum_1/Einsum:output:0pmodel_1/transformer_encoder_3/Encoder-SelfAttentionLayer-1/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:���������P	*
equationabcd,cde->abe�
^model_1/transformer_encoder_3/Encoder-SelfAttentionLayer-1/attention_output/add/ReadVariableOpReadVariableOpgmodel_1_transformer_encoder_3_encoder_selfattentionlayer_1_attention_output_add_readvariableop_resource*
_output_shapes
:	*
dtype0�
Omodel_1/transformer_encoder_3/Encoder-SelfAttentionLayer-1/attention_output/addAddV2bmodel_1/transformer_encoder_3/Encoder-SelfAttentionLayer-1/attention_output/einsum/Einsum:output:0fmodel_1/transformer_encoder_3/Encoder-SelfAttentionLayer-1/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������P	�
=model_1/transformer_encoder_3/Encoder-1st-AdditionLayer-1/addAddV2&model_1/transformer_encoder_3/Cast:y:0Smodel_1/transformer_encoder_3/Encoder-SelfAttentionLayer-1/attention_output/add:z:0*
T0*+
_output_shapes
:���������P	�
Dmodel_1/transformer_encoder_3/Encoder-2nd-NormalizationLayer-1/ShapeShapeAmodel_1/transformer_encoder_3/Encoder-1st-AdditionLayer-1/add:z:0*
T0*
_output_shapes
::���
Rmodel_1/transformer_encoder_3/Encoder-2nd-NormalizationLayer-1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
Tmodel_1/transformer_encoder_3/Encoder-2nd-NormalizationLayer-1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
Tmodel_1/transformer_encoder_3/Encoder-2nd-NormalizationLayer-1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
Lmodel_1/transformer_encoder_3/Encoder-2nd-NormalizationLayer-1/strided_sliceStridedSliceMmodel_1/transformer_encoder_3/Encoder-2nd-NormalizationLayer-1/Shape:output:0[model_1/transformer_encoder_3/Encoder-2nd-NormalizationLayer-1/strided_slice/stack:output:0]model_1/transformer_encoder_3/Encoder-2nd-NormalizationLayer-1/strided_slice/stack_1:output:0]model_1/transformer_encoder_3/Encoder-2nd-NormalizationLayer-1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask�
Dmodel_1/transformer_encoder_3/Encoder-2nd-NormalizationLayer-1/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
Cmodel_1/transformer_encoder_3/Encoder-2nd-NormalizationLayer-1/ProdProdUmodel_1/transformer_encoder_3/Encoder-2nd-NormalizationLayer-1/strided_slice:output:0Mmodel_1/transformer_encoder_3/Encoder-2nd-NormalizationLayer-1/Const:output:0*
T0*
_output_shapes
: �
Tmodel_1/transformer_encoder_3/Encoder-2nd-NormalizationLayer-1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:�
Vmodel_1/transformer_encoder_3/Encoder-2nd-NormalizationLayer-1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: �
Vmodel_1/transformer_encoder_3/Encoder-2nd-NormalizationLayer-1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
Nmodel_1/transformer_encoder_3/Encoder-2nd-NormalizationLayer-1/strided_slice_1StridedSliceMmodel_1/transformer_encoder_3/Encoder-2nd-NormalizationLayer-1/Shape:output:0]model_1/transformer_encoder_3/Encoder-2nd-NormalizationLayer-1/strided_slice_1/stack:output:0_model_1/transformer_encoder_3/Encoder-2nd-NormalizationLayer-1/strided_slice_1/stack_1:output:0_model_1/transformer_encoder_3/Encoder-2nd-NormalizationLayer-1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask�
Fmodel_1/transformer_encoder_3/Encoder-2nd-NormalizationLayer-1/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
Emodel_1/transformer_encoder_3/Encoder-2nd-NormalizationLayer-1/Prod_1ProdWmodel_1/transformer_encoder_3/Encoder-2nd-NormalizationLayer-1/strided_slice_1:output:0Omodel_1/transformer_encoder_3/Encoder-2nd-NormalizationLayer-1/Const_1:output:0*
T0*
_output_shapes
: �
Nmodel_1/transformer_encoder_3/Encoder-2nd-NormalizationLayer-1/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :�
Nmodel_1/transformer_encoder_3/Encoder-2nd-NormalizationLayer-1/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :�
Lmodel_1/transformer_encoder_3/Encoder-2nd-NormalizationLayer-1/Reshape/shapePackWmodel_1/transformer_encoder_3/Encoder-2nd-NormalizationLayer-1/Reshape/shape/0:output:0Lmodel_1/transformer_encoder_3/Encoder-2nd-NormalizationLayer-1/Prod:output:0Nmodel_1/transformer_encoder_3/Encoder-2nd-NormalizationLayer-1/Prod_1:output:0Wmodel_1/transformer_encoder_3/Encoder-2nd-NormalizationLayer-1/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:�
Fmodel_1/transformer_encoder_3/Encoder-2nd-NormalizationLayer-1/ReshapeReshapeAmodel_1/transformer_encoder_3/Encoder-1st-AdditionLayer-1/add:z:0Umodel_1/transformer_encoder_3/Encoder-2nd-NormalizationLayer-1/Reshape/shape:output:0*
T0*8
_output_shapes&
$:"�������������������
Jmodel_1/transformer_encoder_3/Encoder-2nd-NormalizationLayer-1/ones/packedPackLmodel_1/transformer_encoder_3/Encoder-2nd-NormalizationLayer-1/Prod:output:0*
N*
T0*
_output_shapes
:�
Imodel_1/transformer_encoder_3/Encoder-2nd-NormalizationLayer-1/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
Cmodel_1/transformer_encoder_3/Encoder-2nd-NormalizationLayer-1/onesFillSmodel_1/transformer_encoder_3/Encoder-2nd-NormalizationLayer-1/ones/packed:output:0Rmodel_1/transformer_encoder_3/Encoder-2nd-NormalizationLayer-1/ones/Const:output:0*
T0*#
_output_shapes
:����������
Kmodel_1/transformer_encoder_3/Encoder-2nd-NormalizationLayer-1/zeros/packedPackLmodel_1/transformer_encoder_3/Encoder-2nd-NormalizationLayer-1/Prod:output:0*
N*
T0*
_output_shapes
:�
Jmodel_1/transformer_encoder_3/Encoder-2nd-NormalizationLayer-1/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
Dmodel_1/transformer_encoder_3/Encoder-2nd-NormalizationLayer-1/zerosFillTmodel_1/transformer_encoder_3/Encoder-2nd-NormalizationLayer-1/zeros/packed:output:0Smodel_1/transformer_encoder_3/Encoder-2nd-NormalizationLayer-1/zeros/Const:output:0*
T0*#
_output_shapes
:����������
Fmodel_1/transformer_encoder_3/Encoder-2nd-NormalizationLayer-1/Const_2Const*
_output_shapes
: *
dtype0*
valueB �
Fmodel_1/transformer_encoder_3/Encoder-2nd-NormalizationLayer-1/Const_3Const*
_output_shapes
: *
dtype0*
valueB �
Omodel_1/transformer_encoder_3/Encoder-2nd-NormalizationLayer-1/FusedBatchNormV3FusedBatchNormV3Omodel_1/transformer_encoder_3/Encoder-2nd-NormalizationLayer-1/Reshape:output:0Lmodel_1/transformer_encoder_3/Encoder-2nd-NormalizationLayer-1/ones:output:0Mmodel_1/transformer_encoder_3/Encoder-2nd-NormalizationLayer-1/zeros:output:0Omodel_1/transformer_encoder_3/Encoder-2nd-NormalizationLayer-1/Const_2:output:0Omodel_1/transformer_encoder_3/Encoder-2nd-NormalizationLayer-1/Const_3:output:0*
T0*
U0*x
_output_shapesf
d:"������������������:���������:���������:���������:���������:*
data_formatNCHW*
epsilon%o�:�
Hmodel_1/transformer_encoder_3/Encoder-2nd-NormalizationLayer-1/Reshape_1ReshapeSmodel_1/transformer_encoder_3/Encoder-2nd-NormalizationLayer-1/FusedBatchNormV3:y:0Mmodel_1/transformer_encoder_3/Encoder-2nd-NormalizationLayer-1/Shape:output:0*
T0*+
_output_shapes
:���������P	�
Qmodel_1/transformer_encoder_3/Encoder-2nd-NormalizationLayer-1/mul/ReadVariableOpReadVariableOpZmodel_1_transformer_encoder_3_encoder_2nd_normalizationlayer_1_mul_readvariableop_resource*
_output_shapes
:	*
dtype0�
Bmodel_1/transformer_encoder_3/Encoder-2nd-NormalizationLayer-1/mulMulQmodel_1/transformer_encoder_3/Encoder-2nd-NormalizationLayer-1/Reshape_1:output:0Ymodel_1/transformer_encoder_3/Encoder-2nd-NormalizationLayer-1/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������P	�
Qmodel_1/transformer_encoder_3/Encoder-2nd-NormalizationLayer-1/add/ReadVariableOpReadVariableOpZmodel_1_transformer_encoder_3_encoder_2nd_normalizationlayer_1_add_readvariableop_resource*
_output_shapes
:	*
dtype0�
Bmodel_1/transformer_encoder_3/Encoder-2nd-NormalizationLayer-1/addAddV2Fmodel_1/transformer_encoder_3/Encoder-2nd-NormalizationLayer-1/mul:z:0Ymodel_1/transformer_encoder_3/Encoder-2nd-NormalizationLayer-1/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������P	�
Smodel_1/transformer_encoder_3/Encoder-FeedForwardLayer_1_1/Tensordot/ReadVariableOpReadVariableOp\model_1_transformer_encoder_3_encoder_feedforwardlayer_1_1_tensordot_readvariableop_resource*
_output_shapes

:		*
dtype0�
Imodel_1/transformer_encoder_3/Encoder-FeedForwardLayer_1_1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:�
Imodel_1/transformer_encoder_3/Encoder-FeedForwardLayer_1_1/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       �
Jmodel_1/transformer_encoder_3/Encoder-FeedForwardLayer_1_1/Tensordot/ShapeShapeFmodel_1/transformer_encoder_3/Encoder-2nd-NormalizationLayer-1/add:z:0*
T0*
_output_shapes
::���
Rmodel_1/transformer_encoder_3/Encoder-FeedForwardLayer_1_1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Mmodel_1/transformer_encoder_3/Encoder-FeedForwardLayer_1_1/Tensordot/GatherV2GatherV2Smodel_1/transformer_encoder_3/Encoder-FeedForwardLayer_1_1/Tensordot/Shape:output:0Rmodel_1/transformer_encoder_3/Encoder-FeedForwardLayer_1_1/Tensordot/free:output:0[model_1/transformer_encoder_3/Encoder-FeedForwardLayer_1_1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
Tmodel_1/transformer_encoder_3/Encoder-FeedForwardLayer_1_1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Omodel_1/transformer_encoder_3/Encoder-FeedForwardLayer_1_1/Tensordot/GatherV2_1GatherV2Smodel_1/transformer_encoder_3/Encoder-FeedForwardLayer_1_1/Tensordot/Shape:output:0Rmodel_1/transformer_encoder_3/Encoder-FeedForwardLayer_1_1/Tensordot/axes:output:0]model_1/transformer_encoder_3/Encoder-FeedForwardLayer_1_1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
Jmodel_1/transformer_encoder_3/Encoder-FeedForwardLayer_1_1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
Imodel_1/transformer_encoder_3/Encoder-FeedForwardLayer_1_1/Tensordot/ProdProdVmodel_1/transformer_encoder_3/Encoder-FeedForwardLayer_1_1/Tensordot/GatherV2:output:0Smodel_1/transformer_encoder_3/Encoder-FeedForwardLayer_1_1/Tensordot/Const:output:0*
T0*
_output_shapes
: �
Lmodel_1/transformer_encoder_3/Encoder-FeedForwardLayer_1_1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
Kmodel_1/transformer_encoder_3/Encoder-FeedForwardLayer_1_1/Tensordot/Prod_1ProdXmodel_1/transformer_encoder_3/Encoder-FeedForwardLayer_1_1/Tensordot/GatherV2_1:output:0Umodel_1/transformer_encoder_3/Encoder-FeedForwardLayer_1_1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: �
Pmodel_1/transformer_encoder_3/Encoder-FeedForwardLayer_1_1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Kmodel_1/transformer_encoder_3/Encoder-FeedForwardLayer_1_1/Tensordot/concatConcatV2Rmodel_1/transformer_encoder_3/Encoder-FeedForwardLayer_1_1/Tensordot/free:output:0Rmodel_1/transformer_encoder_3/Encoder-FeedForwardLayer_1_1/Tensordot/axes:output:0Ymodel_1/transformer_encoder_3/Encoder-FeedForwardLayer_1_1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
Jmodel_1/transformer_encoder_3/Encoder-FeedForwardLayer_1_1/Tensordot/stackPackRmodel_1/transformer_encoder_3/Encoder-FeedForwardLayer_1_1/Tensordot/Prod:output:0Tmodel_1/transformer_encoder_3/Encoder-FeedForwardLayer_1_1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
Nmodel_1/transformer_encoder_3/Encoder-FeedForwardLayer_1_1/Tensordot/transpose	TransposeFmodel_1/transformer_encoder_3/Encoder-2nd-NormalizationLayer-1/add:z:0Tmodel_1/transformer_encoder_3/Encoder-FeedForwardLayer_1_1/Tensordot/concat:output:0*
T0*+
_output_shapes
:���������P	�
Lmodel_1/transformer_encoder_3/Encoder-FeedForwardLayer_1_1/Tensordot/ReshapeReshapeRmodel_1/transformer_encoder_3/Encoder-FeedForwardLayer_1_1/Tensordot/transpose:y:0Smodel_1/transformer_encoder_3/Encoder-FeedForwardLayer_1_1/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
Kmodel_1/transformer_encoder_3/Encoder-FeedForwardLayer_1_1/Tensordot/MatMulMatMulUmodel_1/transformer_encoder_3/Encoder-FeedForwardLayer_1_1/Tensordot/Reshape:output:0[model_1/transformer_encoder_3/Encoder-FeedForwardLayer_1_1/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������	�
Lmodel_1/transformer_encoder_3/Encoder-FeedForwardLayer_1_1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:	�
Rmodel_1/transformer_encoder_3/Encoder-FeedForwardLayer_1_1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Mmodel_1/transformer_encoder_3/Encoder-FeedForwardLayer_1_1/Tensordot/concat_1ConcatV2Vmodel_1/transformer_encoder_3/Encoder-FeedForwardLayer_1_1/Tensordot/GatherV2:output:0Umodel_1/transformer_encoder_3/Encoder-FeedForwardLayer_1_1/Tensordot/Const_2:output:0[model_1/transformer_encoder_3/Encoder-FeedForwardLayer_1_1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
Dmodel_1/transformer_encoder_3/Encoder-FeedForwardLayer_1_1/TensordotReshapeUmodel_1/transformer_encoder_3/Encoder-FeedForwardLayer_1_1/Tensordot/MatMul:product:0Vmodel_1/transformer_encoder_3/Encoder-FeedForwardLayer_1_1/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������P	�
Qmodel_1/transformer_encoder_3/Encoder-FeedForwardLayer_1_1/BiasAdd/ReadVariableOpReadVariableOpZmodel_1_transformer_encoder_3_encoder_feedforwardlayer_1_1_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype0�
Bmodel_1/transformer_encoder_3/Encoder-FeedForwardLayer_1_1/BiasAddBiasAddMmodel_1/transformer_encoder_3/Encoder-FeedForwardLayer_1_1/Tensordot:output:0Ymodel_1/transformer_encoder_3/Encoder-FeedForwardLayer_1_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������P	�
Emodel_1/transformer_encoder_3/Encoder-FeedForwardLayer_1_1/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
Cmodel_1/transformer_encoder_3/Encoder-FeedForwardLayer_1_1/Gelu/mulMulNmodel_1/transformer_encoder_3/Encoder-FeedForwardLayer_1_1/Gelu/mul/x:output:0Kmodel_1/transformer_encoder_3/Encoder-FeedForwardLayer_1_1/BiasAdd:output:0*
T0*+
_output_shapes
:���������P	�
Fmodel_1/transformer_encoder_3/Encoder-FeedForwardLayer_1_1/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *��?�
Gmodel_1/transformer_encoder_3/Encoder-FeedForwardLayer_1_1/Gelu/truedivRealDivKmodel_1/transformer_encoder_3/Encoder-FeedForwardLayer_1_1/BiasAdd:output:0Omodel_1/transformer_encoder_3/Encoder-FeedForwardLayer_1_1/Gelu/Cast/x:output:0*
T0*+
_output_shapes
:���������P	�
Cmodel_1/transformer_encoder_3/Encoder-FeedForwardLayer_1_1/Gelu/ErfErfKmodel_1/transformer_encoder_3/Encoder-FeedForwardLayer_1_1/Gelu/truediv:z:0*
T0*+
_output_shapes
:���������P	�
Emodel_1/transformer_encoder_3/Encoder-FeedForwardLayer_1_1/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
Cmodel_1/transformer_encoder_3/Encoder-FeedForwardLayer_1_1/Gelu/addAddV2Nmodel_1/transformer_encoder_3/Encoder-FeedForwardLayer_1_1/Gelu/add/x:output:0Gmodel_1/transformer_encoder_3/Encoder-FeedForwardLayer_1_1/Gelu/Erf:y:0*
T0*+
_output_shapes
:���������P	�
Emodel_1/transformer_encoder_3/Encoder-FeedForwardLayer_1_1/Gelu/mul_1MulGmodel_1/transformer_encoder_3/Encoder-FeedForwardLayer_1_1/Gelu/mul:z:0Gmodel_1/transformer_encoder_3/Encoder-FeedForwardLayer_1_1/Gelu/add:z:0*
T0*+
_output_shapes
:���������P	�
Smodel_1/transformer_encoder_3/Encoder-FeedForwardLayer_2_1/Tensordot/ReadVariableOpReadVariableOp\model_1_transformer_encoder_3_encoder_feedforwardlayer_2_1_tensordot_readvariableop_resource*
_output_shapes

:		*
dtype0�
Imodel_1/transformer_encoder_3/Encoder-FeedForwardLayer_2_1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:�
Imodel_1/transformer_encoder_3/Encoder-FeedForwardLayer_2_1/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       �
Jmodel_1/transformer_encoder_3/Encoder-FeedForwardLayer_2_1/Tensordot/ShapeShapeImodel_1/transformer_encoder_3/Encoder-FeedForwardLayer_1_1/Gelu/mul_1:z:0*
T0*
_output_shapes
::���
Rmodel_1/transformer_encoder_3/Encoder-FeedForwardLayer_2_1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Mmodel_1/transformer_encoder_3/Encoder-FeedForwardLayer_2_1/Tensordot/GatherV2GatherV2Smodel_1/transformer_encoder_3/Encoder-FeedForwardLayer_2_1/Tensordot/Shape:output:0Rmodel_1/transformer_encoder_3/Encoder-FeedForwardLayer_2_1/Tensordot/free:output:0[model_1/transformer_encoder_3/Encoder-FeedForwardLayer_2_1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
Tmodel_1/transformer_encoder_3/Encoder-FeedForwardLayer_2_1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Omodel_1/transformer_encoder_3/Encoder-FeedForwardLayer_2_1/Tensordot/GatherV2_1GatherV2Smodel_1/transformer_encoder_3/Encoder-FeedForwardLayer_2_1/Tensordot/Shape:output:0Rmodel_1/transformer_encoder_3/Encoder-FeedForwardLayer_2_1/Tensordot/axes:output:0]model_1/transformer_encoder_3/Encoder-FeedForwardLayer_2_1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
Jmodel_1/transformer_encoder_3/Encoder-FeedForwardLayer_2_1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
Imodel_1/transformer_encoder_3/Encoder-FeedForwardLayer_2_1/Tensordot/ProdProdVmodel_1/transformer_encoder_3/Encoder-FeedForwardLayer_2_1/Tensordot/GatherV2:output:0Smodel_1/transformer_encoder_3/Encoder-FeedForwardLayer_2_1/Tensordot/Const:output:0*
T0*
_output_shapes
: �
Lmodel_1/transformer_encoder_3/Encoder-FeedForwardLayer_2_1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
Kmodel_1/transformer_encoder_3/Encoder-FeedForwardLayer_2_1/Tensordot/Prod_1ProdXmodel_1/transformer_encoder_3/Encoder-FeedForwardLayer_2_1/Tensordot/GatherV2_1:output:0Umodel_1/transformer_encoder_3/Encoder-FeedForwardLayer_2_1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: �
Pmodel_1/transformer_encoder_3/Encoder-FeedForwardLayer_2_1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Kmodel_1/transformer_encoder_3/Encoder-FeedForwardLayer_2_1/Tensordot/concatConcatV2Rmodel_1/transformer_encoder_3/Encoder-FeedForwardLayer_2_1/Tensordot/free:output:0Rmodel_1/transformer_encoder_3/Encoder-FeedForwardLayer_2_1/Tensordot/axes:output:0Ymodel_1/transformer_encoder_3/Encoder-FeedForwardLayer_2_1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
Jmodel_1/transformer_encoder_3/Encoder-FeedForwardLayer_2_1/Tensordot/stackPackRmodel_1/transformer_encoder_3/Encoder-FeedForwardLayer_2_1/Tensordot/Prod:output:0Tmodel_1/transformer_encoder_3/Encoder-FeedForwardLayer_2_1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
Nmodel_1/transformer_encoder_3/Encoder-FeedForwardLayer_2_1/Tensordot/transpose	TransposeImodel_1/transformer_encoder_3/Encoder-FeedForwardLayer_1_1/Gelu/mul_1:z:0Tmodel_1/transformer_encoder_3/Encoder-FeedForwardLayer_2_1/Tensordot/concat:output:0*
T0*+
_output_shapes
:���������P	�
Lmodel_1/transformer_encoder_3/Encoder-FeedForwardLayer_2_1/Tensordot/ReshapeReshapeRmodel_1/transformer_encoder_3/Encoder-FeedForwardLayer_2_1/Tensordot/transpose:y:0Smodel_1/transformer_encoder_3/Encoder-FeedForwardLayer_2_1/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
Kmodel_1/transformer_encoder_3/Encoder-FeedForwardLayer_2_1/Tensordot/MatMulMatMulUmodel_1/transformer_encoder_3/Encoder-FeedForwardLayer_2_1/Tensordot/Reshape:output:0[model_1/transformer_encoder_3/Encoder-FeedForwardLayer_2_1/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������	�
Lmodel_1/transformer_encoder_3/Encoder-FeedForwardLayer_2_1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:	�
Rmodel_1/transformer_encoder_3/Encoder-FeedForwardLayer_2_1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Mmodel_1/transformer_encoder_3/Encoder-FeedForwardLayer_2_1/Tensordot/concat_1ConcatV2Vmodel_1/transformer_encoder_3/Encoder-FeedForwardLayer_2_1/Tensordot/GatherV2:output:0Umodel_1/transformer_encoder_3/Encoder-FeedForwardLayer_2_1/Tensordot/Const_2:output:0[model_1/transformer_encoder_3/Encoder-FeedForwardLayer_2_1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
Dmodel_1/transformer_encoder_3/Encoder-FeedForwardLayer_2_1/TensordotReshapeUmodel_1/transformer_encoder_3/Encoder-FeedForwardLayer_2_1/Tensordot/MatMul:product:0Vmodel_1/transformer_encoder_3/Encoder-FeedForwardLayer_2_1/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������P	�
Qmodel_1/transformer_encoder_3/Encoder-FeedForwardLayer_2_1/BiasAdd/ReadVariableOpReadVariableOpZmodel_1_transformer_encoder_3_encoder_feedforwardlayer_2_1_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype0�
Bmodel_1/transformer_encoder_3/Encoder-FeedForwardLayer_2_1/BiasAddBiasAddMmodel_1/transformer_encoder_3/Encoder-FeedForwardLayer_2_1/Tensordot:output:0Ymodel_1/transformer_encoder_3/Encoder-FeedForwardLayer_2_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������P	�
0model_1/transformer_encoder_3/dropout_3/IdentityIdentityKmodel_1/transformer_encoder_3/Encoder-FeedForwardLayer_2_1/BiasAdd:output:0*
T0*+
_output_shapes
:���������P	�
=model_1/transformer_encoder_3/Encoder-2nd-AdditionLayer-1/addAddV2Amodel_1/transformer_encoder_3/Encoder-1st-AdditionLayer-1/add:z:09model_1/transformer_encoder_3/dropout_3/Identity:output:0*
T0*+
_output_shapes
:���������P	�
Dmodel_1/transformer_encoder_4/Encoder-1st-NormalizationLayer-2/ShapeShapeAmodel_1/transformer_encoder_3/Encoder-2nd-AdditionLayer-1/add:z:0*
T0*
_output_shapes
::���
Rmodel_1/transformer_encoder_4/Encoder-1st-NormalizationLayer-2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
Tmodel_1/transformer_encoder_4/Encoder-1st-NormalizationLayer-2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
Tmodel_1/transformer_encoder_4/Encoder-1st-NormalizationLayer-2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
Lmodel_1/transformer_encoder_4/Encoder-1st-NormalizationLayer-2/strided_sliceStridedSliceMmodel_1/transformer_encoder_4/Encoder-1st-NormalizationLayer-2/Shape:output:0[model_1/transformer_encoder_4/Encoder-1st-NormalizationLayer-2/strided_slice/stack:output:0]model_1/transformer_encoder_4/Encoder-1st-NormalizationLayer-2/strided_slice/stack_1:output:0]model_1/transformer_encoder_4/Encoder-1st-NormalizationLayer-2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask�
Dmodel_1/transformer_encoder_4/Encoder-1st-NormalizationLayer-2/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
Cmodel_1/transformer_encoder_4/Encoder-1st-NormalizationLayer-2/ProdProdUmodel_1/transformer_encoder_4/Encoder-1st-NormalizationLayer-2/strided_slice:output:0Mmodel_1/transformer_encoder_4/Encoder-1st-NormalizationLayer-2/Const:output:0*
T0*
_output_shapes
: �
Tmodel_1/transformer_encoder_4/Encoder-1st-NormalizationLayer-2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:�
Vmodel_1/transformer_encoder_4/Encoder-1st-NormalizationLayer-2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: �
Vmodel_1/transformer_encoder_4/Encoder-1st-NormalizationLayer-2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
Nmodel_1/transformer_encoder_4/Encoder-1st-NormalizationLayer-2/strided_slice_1StridedSliceMmodel_1/transformer_encoder_4/Encoder-1st-NormalizationLayer-2/Shape:output:0]model_1/transformer_encoder_4/Encoder-1st-NormalizationLayer-2/strided_slice_1/stack:output:0_model_1/transformer_encoder_4/Encoder-1st-NormalizationLayer-2/strided_slice_1/stack_1:output:0_model_1/transformer_encoder_4/Encoder-1st-NormalizationLayer-2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask�
Fmodel_1/transformer_encoder_4/Encoder-1st-NormalizationLayer-2/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
Emodel_1/transformer_encoder_4/Encoder-1st-NormalizationLayer-2/Prod_1ProdWmodel_1/transformer_encoder_4/Encoder-1st-NormalizationLayer-2/strided_slice_1:output:0Omodel_1/transformer_encoder_4/Encoder-1st-NormalizationLayer-2/Const_1:output:0*
T0*
_output_shapes
: �
Nmodel_1/transformer_encoder_4/Encoder-1st-NormalizationLayer-2/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :�
Nmodel_1/transformer_encoder_4/Encoder-1st-NormalizationLayer-2/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :�
Lmodel_1/transformer_encoder_4/Encoder-1st-NormalizationLayer-2/Reshape/shapePackWmodel_1/transformer_encoder_4/Encoder-1st-NormalizationLayer-2/Reshape/shape/0:output:0Lmodel_1/transformer_encoder_4/Encoder-1st-NormalizationLayer-2/Prod:output:0Nmodel_1/transformer_encoder_4/Encoder-1st-NormalizationLayer-2/Prod_1:output:0Wmodel_1/transformer_encoder_4/Encoder-1st-NormalizationLayer-2/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:�
Fmodel_1/transformer_encoder_4/Encoder-1st-NormalizationLayer-2/ReshapeReshapeAmodel_1/transformer_encoder_3/Encoder-2nd-AdditionLayer-1/add:z:0Umodel_1/transformer_encoder_4/Encoder-1st-NormalizationLayer-2/Reshape/shape:output:0*
T0*8
_output_shapes&
$:"�������������������
Jmodel_1/transformer_encoder_4/Encoder-1st-NormalizationLayer-2/ones/packedPackLmodel_1/transformer_encoder_4/Encoder-1st-NormalizationLayer-2/Prod:output:0*
N*
T0*
_output_shapes
:�
Imodel_1/transformer_encoder_4/Encoder-1st-NormalizationLayer-2/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
Cmodel_1/transformer_encoder_4/Encoder-1st-NormalizationLayer-2/onesFillSmodel_1/transformer_encoder_4/Encoder-1st-NormalizationLayer-2/ones/packed:output:0Rmodel_1/transformer_encoder_4/Encoder-1st-NormalizationLayer-2/ones/Const:output:0*
T0*#
_output_shapes
:����������
Kmodel_1/transformer_encoder_4/Encoder-1st-NormalizationLayer-2/zeros/packedPackLmodel_1/transformer_encoder_4/Encoder-1st-NormalizationLayer-2/Prod:output:0*
N*
T0*
_output_shapes
:�
Jmodel_1/transformer_encoder_4/Encoder-1st-NormalizationLayer-2/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
Dmodel_1/transformer_encoder_4/Encoder-1st-NormalizationLayer-2/zerosFillTmodel_1/transformer_encoder_4/Encoder-1st-NormalizationLayer-2/zeros/packed:output:0Smodel_1/transformer_encoder_4/Encoder-1st-NormalizationLayer-2/zeros/Const:output:0*
T0*#
_output_shapes
:����������
Fmodel_1/transformer_encoder_4/Encoder-1st-NormalizationLayer-2/Const_2Const*
_output_shapes
: *
dtype0*
valueB �
Fmodel_1/transformer_encoder_4/Encoder-1st-NormalizationLayer-2/Const_3Const*
_output_shapes
: *
dtype0*
valueB �
Omodel_1/transformer_encoder_4/Encoder-1st-NormalizationLayer-2/FusedBatchNormV3FusedBatchNormV3Omodel_1/transformer_encoder_4/Encoder-1st-NormalizationLayer-2/Reshape:output:0Lmodel_1/transformer_encoder_4/Encoder-1st-NormalizationLayer-2/ones:output:0Mmodel_1/transformer_encoder_4/Encoder-1st-NormalizationLayer-2/zeros:output:0Omodel_1/transformer_encoder_4/Encoder-1st-NormalizationLayer-2/Const_2:output:0Omodel_1/transformer_encoder_4/Encoder-1st-NormalizationLayer-2/Const_3:output:0*
T0*
U0*x
_output_shapesf
d:"������������������:���������:���������:���������:���������:*
data_formatNCHW*
epsilon%o�:�
Hmodel_1/transformer_encoder_4/Encoder-1st-NormalizationLayer-2/Reshape_1ReshapeSmodel_1/transformer_encoder_4/Encoder-1st-NormalizationLayer-2/FusedBatchNormV3:y:0Mmodel_1/transformer_encoder_4/Encoder-1st-NormalizationLayer-2/Shape:output:0*
T0*+
_output_shapes
:���������P	�
Qmodel_1/transformer_encoder_4/Encoder-1st-NormalizationLayer-2/mul/ReadVariableOpReadVariableOpZmodel_1_transformer_encoder_4_encoder_1st_normalizationlayer_2_mul_readvariableop_resource*
_output_shapes
:	*
dtype0�
Bmodel_1/transformer_encoder_4/Encoder-1st-NormalizationLayer-2/mulMulQmodel_1/transformer_encoder_4/Encoder-1st-NormalizationLayer-2/Reshape_1:output:0Ymodel_1/transformer_encoder_4/Encoder-1st-NormalizationLayer-2/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������P	�
Qmodel_1/transformer_encoder_4/Encoder-1st-NormalizationLayer-2/add/ReadVariableOpReadVariableOpZmodel_1_transformer_encoder_4_encoder_1st_normalizationlayer_2_add_readvariableop_resource*
_output_shapes
:	*
dtype0�
Bmodel_1/transformer_encoder_4/Encoder-1st-NormalizationLayer-2/addAddV2Fmodel_1/transformer_encoder_4/Encoder-1st-NormalizationLayer-2/mul:z:0Ymodel_1/transformer_encoder_4/Encoder-1st-NormalizationLayer-2/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������P	�
]model_1/transformer_encoder_4/Encoder-SelfAttentionLayer-2/query/einsum/Einsum/ReadVariableOpReadVariableOpfmodel_1_transformer_encoder_4_encoder_selfattentionlayer_2_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:	*
dtype0�
Nmodel_1/transformer_encoder_4/Encoder-SelfAttentionLayer-2/query/einsum/EinsumEinsumFmodel_1/transformer_encoder_4/Encoder-1st-NormalizationLayer-2/add:z:0emodel_1/transformer_encoder_4/Encoder-SelfAttentionLayer-2/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������P*
equationabc,cde->abde�
Smodel_1/transformer_encoder_4/Encoder-SelfAttentionLayer-2/query/add/ReadVariableOpReadVariableOp\model_1_transformer_encoder_4_encoder_selfattentionlayer_2_query_add_readvariableop_resource*
_output_shapes

:*
dtype0�
Dmodel_1/transformer_encoder_4/Encoder-SelfAttentionLayer-2/query/addAddV2Wmodel_1/transformer_encoder_4/Encoder-SelfAttentionLayer-2/query/einsum/Einsum:output:0[model_1/transformer_encoder_4/Encoder-SelfAttentionLayer-2/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������P�
[model_1/transformer_encoder_4/Encoder-SelfAttentionLayer-2/key/einsum/Einsum/ReadVariableOpReadVariableOpdmodel_1_transformer_encoder_4_encoder_selfattentionlayer_2_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:	*
dtype0�
Lmodel_1/transformer_encoder_4/Encoder-SelfAttentionLayer-2/key/einsum/EinsumEinsumFmodel_1/transformer_encoder_4/Encoder-1st-NormalizationLayer-2/add:z:0cmodel_1/transformer_encoder_4/Encoder-SelfAttentionLayer-2/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������P*
equationabc,cde->abde�
Qmodel_1/transformer_encoder_4/Encoder-SelfAttentionLayer-2/key/add/ReadVariableOpReadVariableOpZmodel_1_transformer_encoder_4_encoder_selfattentionlayer_2_key_add_readvariableop_resource*
_output_shapes

:*
dtype0�
Bmodel_1/transformer_encoder_4/Encoder-SelfAttentionLayer-2/key/addAddV2Umodel_1/transformer_encoder_4/Encoder-SelfAttentionLayer-2/key/einsum/Einsum:output:0Ymodel_1/transformer_encoder_4/Encoder-SelfAttentionLayer-2/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������P�
]model_1/transformer_encoder_4/Encoder-SelfAttentionLayer-2/value/einsum/Einsum/ReadVariableOpReadVariableOpfmodel_1_transformer_encoder_4_encoder_selfattentionlayer_2_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:	*
dtype0�
Nmodel_1/transformer_encoder_4/Encoder-SelfAttentionLayer-2/value/einsum/EinsumEinsumFmodel_1/transformer_encoder_4/Encoder-1st-NormalizationLayer-2/add:z:0emodel_1/transformer_encoder_4/Encoder-SelfAttentionLayer-2/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������P*
equationabc,cde->abde�
Smodel_1/transformer_encoder_4/Encoder-SelfAttentionLayer-2/value/add/ReadVariableOpReadVariableOp\model_1_transformer_encoder_4_encoder_selfattentionlayer_2_value_add_readvariableop_resource*
_output_shapes

:*
dtype0�
Dmodel_1/transformer_encoder_4/Encoder-SelfAttentionLayer-2/value/addAddV2Wmodel_1/transformer_encoder_4/Encoder-SelfAttentionLayer-2/value/einsum/Einsum:output:0[model_1/transformer_encoder_4/Encoder-SelfAttentionLayer-2/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������P�
@model_1/transformer_encoder_4/Encoder-SelfAttentionLayer-2/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *:�?�
>model_1/transformer_encoder_4/Encoder-SelfAttentionLayer-2/MulMulHmodel_1/transformer_encoder_4/Encoder-SelfAttentionLayer-2/query/add:z:0Imodel_1/transformer_encoder_4/Encoder-SelfAttentionLayer-2/Mul/y:output:0*
T0*/
_output_shapes
:���������P�
Hmodel_1/transformer_encoder_4/Encoder-SelfAttentionLayer-2/einsum/EinsumEinsumFmodel_1/transformer_encoder_4/Encoder-SelfAttentionLayer-2/key/add:z:0Bmodel_1/transformer_encoder_4/Encoder-SelfAttentionLayer-2/Mul:z:0*
N*
T0*/
_output_shapes
:���������PP*
equationaecd,abcd->acbe�
Jmodel_1/transformer_encoder_4/Encoder-SelfAttentionLayer-2/softmax/SoftmaxSoftmaxQmodel_1/transformer_encoder_4/Encoder-SelfAttentionLayer-2/einsum/Einsum:output:0*
T0*/
_output_shapes
:���������PP�
Kmodel_1/transformer_encoder_4/Encoder-SelfAttentionLayer-2/dropout/IdentityIdentityTmodel_1/transformer_encoder_4/Encoder-SelfAttentionLayer-2/softmax/Softmax:softmax:0*
T0*/
_output_shapes
:���������PP�
Jmodel_1/transformer_encoder_4/Encoder-SelfAttentionLayer-2/einsum_1/EinsumEinsumTmodel_1/transformer_encoder_4/Encoder-SelfAttentionLayer-2/dropout/Identity:output:0Hmodel_1/transformer_encoder_4/Encoder-SelfAttentionLayer-2/value/add:z:0*
N*
T0*/
_output_shapes
:���������P*
equationacbe,aecd->abcd�
hmodel_1/transformer_encoder_4/Encoder-SelfAttentionLayer-2/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpqmodel_1_transformer_encoder_4_encoder_selfattentionlayer_2_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:	*
dtype0�
Ymodel_1/transformer_encoder_4/Encoder-SelfAttentionLayer-2/attention_output/einsum/EinsumEinsumSmodel_1/transformer_encoder_4/Encoder-SelfAttentionLayer-2/einsum_1/Einsum:output:0pmodel_1/transformer_encoder_4/Encoder-SelfAttentionLayer-2/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:���������P	*
equationabcd,cde->abe�
^model_1/transformer_encoder_4/Encoder-SelfAttentionLayer-2/attention_output/add/ReadVariableOpReadVariableOpgmodel_1_transformer_encoder_4_encoder_selfattentionlayer_2_attention_output_add_readvariableop_resource*
_output_shapes
:	*
dtype0�
Omodel_1/transformer_encoder_4/Encoder-SelfAttentionLayer-2/attention_output/addAddV2bmodel_1/transformer_encoder_4/Encoder-SelfAttentionLayer-2/attention_output/einsum/Einsum:output:0fmodel_1/transformer_encoder_4/Encoder-SelfAttentionLayer-2/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������P	�
=model_1/transformer_encoder_4/Encoder-1st-AdditionLayer-2/addAddV2Amodel_1/transformer_encoder_3/Encoder-2nd-AdditionLayer-1/add:z:0Smodel_1/transformer_encoder_4/Encoder-SelfAttentionLayer-2/attention_output/add:z:0*
T0*+
_output_shapes
:���������P	�
Dmodel_1/transformer_encoder_4/Encoder-2nd-NormalizationLayer-2/ShapeShapeAmodel_1/transformer_encoder_4/Encoder-1st-AdditionLayer-2/add:z:0*
T0*
_output_shapes
::���
Rmodel_1/transformer_encoder_4/Encoder-2nd-NormalizationLayer-2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
Tmodel_1/transformer_encoder_4/Encoder-2nd-NormalizationLayer-2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
Tmodel_1/transformer_encoder_4/Encoder-2nd-NormalizationLayer-2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
Lmodel_1/transformer_encoder_4/Encoder-2nd-NormalizationLayer-2/strided_sliceStridedSliceMmodel_1/transformer_encoder_4/Encoder-2nd-NormalizationLayer-2/Shape:output:0[model_1/transformer_encoder_4/Encoder-2nd-NormalizationLayer-2/strided_slice/stack:output:0]model_1/transformer_encoder_4/Encoder-2nd-NormalizationLayer-2/strided_slice/stack_1:output:0]model_1/transformer_encoder_4/Encoder-2nd-NormalizationLayer-2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask�
Dmodel_1/transformer_encoder_4/Encoder-2nd-NormalizationLayer-2/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
Cmodel_1/transformer_encoder_4/Encoder-2nd-NormalizationLayer-2/ProdProdUmodel_1/transformer_encoder_4/Encoder-2nd-NormalizationLayer-2/strided_slice:output:0Mmodel_1/transformer_encoder_4/Encoder-2nd-NormalizationLayer-2/Const:output:0*
T0*
_output_shapes
: �
Tmodel_1/transformer_encoder_4/Encoder-2nd-NormalizationLayer-2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:�
Vmodel_1/transformer_encoder_4/Encoder-2nd-NormalizationLayer-2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: �
Vmodel_1/transformer_encoder_4/Encoder-2nd-NormalizationLayer-2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
Nmodel_1/transformer_encoder_4/Encoder-2nd-NormalizationLayer-2/strided_slice_1StridedSliceMmodel_1/transformer_encoder_4/Encoder-2nd-NormalizationLayer-2/Shape:output:0]model_1/transformer_encoder_4/Encoder-2nd-NormalizationLayer-2/strided_slice_1/stack:output:0_model_1/transformer_encoder_4/Encoder-2nd-NormalizationLayer-2/strided_slice_1/stack_1:output:0_model_1/transformer_encoder_4/Encoder-2nd-NormalizationLayer-2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask�
Fmodel_1/transformer_encoder_4/Encoder-2nd-NormalizationLayer-2/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
Emodel_1/transformer_encoder_4/Encoder-2nd-NormalizationLayer-2/Prod_1ProdWmodel_1/transformer_encoder_4/Encoder-2nd-NormalizationLayer-2/strided_slice_1:output:0Omodel_1/transformer_encoder_4/Encoder-2nd-NormalizationLayer-2/Const_1:output:0*
T0*
_output_shapes
: �
Nmodel_1/transformer_encoder_4/Encoder-2nd-NormalizationLayer-2/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :�
Nmodel_1/transformer_encoder_4/Encoder-2nd-NormalizationLayer-2/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :�
Lmodel_1/transformer_encoder_4/Encoder-2nd-NormalizationLayer-2/Reshape/shapePackWmodel_1/transformer_encoder_4/Encoder-2nd-NormalizationLayer-2/Reshape/shape/0:output:0Lmodel_1/transformer_encoder_4/Encoder-2nd-NormalizationLayer-2/Prod:output:0Nmodel_1/transformer_encoder_4/Encoder-2nd-NormalizationLayer-2/Prod_1:output:0Wmodel_1/transformer_encoder_4/Encoder-2nd-NormalizationLayer-2/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:�
Fmodel_1/transformer_encoder_4/Encoder-2nd-NormalizationLayer-2/ReshapeReshapeAmodel_1/transformer_encoder_4/Encoder-1st-AdditionLayer-2/add:z:0Umodel_1/transformer_encoder_4/Encoder-2nd-NormalizationLayer-2/Reshape/shape:output:0*
T0*8
_output_shapes&
$:"�������������������
Jmodel_1/transformer_encoder_4/Encoder-2nd-NormalizationLayer-2/ones/packedPackLmodel_1/transformer_encoder_4/Encoder-2nd-NormalizationLayer-2/Prod:output:0*
N*
T0*
_output_shapes
:�
Imodel_1/transformer_encoder_4/Encoder-2nd-NormalizationLayer-2/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
Cmodel_1/transformer_encoder_4/Encoder-2nd-NormalizationLayer-2/onesFillSmodel_1/transformer_encoder_4/Encoder-2nd-NormalizationLayer-2/ones/packed:output:0Rmodel_1/transformer_encoder_4/Encoder-2nd-NormalizationLayer-2/ones/Const:output:0*
T0*#
_output_shapes
:����������
Kmodel_1/transformer_encoder_4/Encoder-2nd-NormalizationLayer-2/zeros/packedPackLmodel_1/transformer_encoder_4/Encoder-2nd-NormalizationLayer-2/Prod:output:0*
N*
T0*
_output_shapes
:�
Jmodel_1/transformer_encoder_4/Encoder-2nd-NormalizationLayer-2/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
Dmodel_1/transformer_encoder_4/Encoder-2nd-NormalizationLayer-2/zerosFillTmodel_1/transformer_encoder_4/Encoder-2nd-NormalizationLayer-2/zeros/packed:output:0Smodel_1/transformer_encoder_4/Encoder-2nd-NormalizationLayer-2/zeros/Const:output:0*
T0*#
_output_shapes
:����������
Fmodel_1/transformer_encoder_4/Encoder-2nd-NormalizationLayer-2/Const_2Const*
_output_shapes
: *
dtype0*
valueB �
Fmodel_1/transformer_encoder_4/Encoder-2nd-NormalizationLayer-2/Const_3Const*
_output_shapes
: *
dtype0*
valueB �
Omodel_1/transformer_encoder_4/Encoder-2nd-NormalizationLayer-2/FusedBatchNormV3FusedBatchNormV3Omodel_1/transformer_encoder_4/Encoder-2nd-NormalizationLayer-2/Reshape:output:0Lmodel_1/transformer_encoder_4/Encoder-2nd-NormalizationLayer-2/ones:output:0Mmodel_1/transformer_encoder_4/Encoder-2nd-NormalizationLayer-2/zeros:output:0Omodel_1/transformer_encoder_4/Encoder-2nd-NormalizationLayer-2/Const_2:output:0Omodel_1/transformer_encoder_4/Encoder-2nd-NormalizationLayer-2/Const_3:output:0*
T0*
U0*x
_output_shapesf
d:"������������������:���������:���������:���������:���������:*
data_formatNCHW*
epsilon%o�:�
Hmodel_1/transformer_encoder_4/Encoder-2nd-NormalizationLayer-2/Reshape_1ReshapeSmodel_1/transformer_encoder_4/Encoder-2nd-NormalizationLayer-2/FusedBatchNormV3:y:0Mmodel_1/transformer_encoder_4/Encoder-2nd-NormalizationLayer-2/Shape:output:0*
T0*+
_output_shapes
:���������P	�
Qmodel_1/transformer_encoder_4/Encoder-2nd-NormalizationLayer-2/mul/ReadVariableOpReadVariableOpZmodel_1_transformer_encoder_4_encoder_2nd_normalizationlayer_2_mul_readvariableop_resource*
_output_shapes
:	*
dtype0�
Bmodel_1/transformer_encoder_4/Encoder-2nd-NormalizationLayer-2/mulMulQmodel_1/transformer_encoder_4/Encoder-2nd-NormalizationLayer-2/Reshape_1:output:0Ymodel_1/transformer_encoder_4/Encoder-2nd-NormalizationLayer-2/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������P	�
Qmodel_1/transformer_encoder_4/Encoder-2nd-NormalizationLayer-2/add/ReadVariableOpReadVariableOpZmodel_1_transformer_encoder_4_encoder_2nd_normalizationlayer_2_add_readvariableop_resource*
_output_shapes
:	*
dtype0�
Bmodel_1/transformer_encoder_4/Encoder-2nd-NormalizationLayer-2/addAddV2Fmodel_1/transformer_encoder_4/Encoder-2nd-NormalizationLayer-2/mul:z:0Ymodel_1/transformer_encoder_4/Encoder-2nd-NormalizationLayer-2/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������P	�
Smodel_1/transformer_encoder_4/Encoder-FeedForwardLayer_1_2/Tensordot/ReadVariableOpReadVariableOp\model_1_transformer_encoder_4_encoder_feedforwardlayer_1_2_tensordot_readvariableop_resource*
_output_shapes

:		*
dtype0�
Imodel_1/transformer_encoder_4/Encoder-FeedForwardLayer_1_2/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:�
Imodel_1/transformer_encoder_4/Encoder-FeedForwardLayer_1_2/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       �
Jmodel_1/transformer_encoder_4/Encoder-FeedForwardLayer_1_2/Tensordot/ShapeShapeFmodel_1/transformer_encoder_4/Encoder-2nd-NormalizationLayer-2/add:z:0*
T0*
_output_shapes
::���
Rmodel_1/transformer_encoder_4/Encoder-FeedForwardLayer_1_2/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Mmodel_1/transformer_encoder_4/Encoder-FeedForwardLayer_1_2/Tensordot/GatherV2GatherV2Smodel_1/transformer_encoder_4/Encoder-FeedForwardLayer_1_2/Tensordot/Shape:output:0Rmodel_1/transformer_encoder_4/Encoder-FeedForwardLayer_1_2/Tensordot/free:output:0[model_1/transformer_encoder_4/Encoder-FeedForwardLayer_1_2/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
Tmodel_1/transformer_encoder_4/Encoder-FeedForwardLayer_1_2/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Omodel_1/transformer_encoder_4/Encoder-FeedForwardLayer_1_2/Tensordot/GatherV2_1GatherV2Smodel_1/transformer_encoder_4/Encoder-FeedForwardLayer_1_2/Tensordot/Shape:output:0Rmodel_1/transformer_encoder_4/Encoder-FeedForwardLayer_1_2/Tensordot/axes:output:0]model_1/transformer_encoder_4/Encoder-FeedForwardLayer_1_2/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
Jmodel_1/transformer_encoder_4/Encoder-FeedForwardLayer_1_2/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
Imodel_1/transformer_encoder_4/Encoder-FeedForwardLayer_1_2/Tensordot/ProdProdVmodel_1/transformer_encoder_4/Encoder-FeedForwardLayer_1_2/Tensordot/GatherV2:output:0Smodel_1/transformer_encoder_4/Encoder-FeedForwardLayer_1_2/Tensordot/Const:output:0*
T0*
_output_shapes
: �
Lmodel_1/transformer_encoder_4/Encoder-FeedForwardLayer_1_2/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
Kmodel_1/transformer_encoder_4/Encoder-FeedForwardLayer_1_2/Tensordot/Prod_1ProdXmodel_1/transformer_encoder_4/Encoder-FeedForwardLayer_1_2/Tensordot/GatherV2_1:output:0Umodel_1/transformer_encoder_4/Encoder-FeedForwardLayer_1_2/Tensordot/Const_1:output:0*
T0*
_output_shapes
: �
Pmodel_1/transformer_encoder_4/Encoder-FeedForwardLayer_1_2/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Kmodel_1/transformer_encoder_4/Encoder-FeedForwardLayer_1_2/Tensordot/concatConcatV2Rmodel_1/transformer_encoder_4/Encoder-FeedForwardLayer_1_2/Tensordot/free:output:0Rmodel_1/transformer_encoder_4/Encoder-FeedForwardLayer_1_2/Tensordot/axes:output:0Ymodel_1/transformer_encoder_4/Encoder-FeedForwardLayer_1_2/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
Jmodel_1/transformer_encoder_4/Encoder-FeedForwardLayer_1_2/Tensordot/stackPackRmodel_1/transformer_encoder_4/Encoder-FeedForwardLayer_1_2/Tensordot/Prod:output:0Tmodel_1/transformer_encoder_4/Encoder-FeedForwardLayer_1_2/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
Nmodel_1/transformer_encoder_4/Encoder-FeedForwardLayer_1_2/Tensordot/transpose	TransposeFmodel_1/transformer_encoder_4/Encoder-2nd-NormalizationLayer-2/add:z:0Tmodel_1/transformer_encoder_4/Encoder-FeedForwardLayer_1_2/Tensordot/concat:output:0*
T0*+
_output_shapes
:���������P	�
Lmodel_1/transformer_encoder_4/Encoder-FeedForwardLayer_1_2/Tensordot/ReshapeReshapeRmodel_1/transformer_encoder_4/Encoder-FeedForwardLayer_1_2/Tensordot/transpose:y:0Smodel_1/transformer_encoder_4/Encoder-FeedForwardLayer_1_2/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
Kmodel_1/transformer_encoder_4/Encoder-FeedForwardLayer_1_2/Tensordot/MatMulMatMulUmodel_1/transformer_encoder_4/Encoder-FeedForwardLayer_1_2/Tensordot/Reshape:output:0[model_1/transformer_encoder_4/Encoder-FeedForwardLayer_1_2/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������	�
Lmodel_1/transformer_encoder_4/Encoder-FeedForwardLayer_1_2/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:	�
Rmodel_1/transformer_encoder_4/Encoder-FeedForwardLayer_1_2/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Mmodel_1/transformer_encoder_4/Encoder-FeedForwardLayer_1_2/Tensordot/concat_1ConcatV2Vmodel_1/transformer_encoder_4/Encoder-FeedForwardLayer_1_2/Tensordot/GatherV2:output:0Umodel_1/transformer_encoder_4/Encoder-FeedForwardLayer_1_2/Tensordot/Const_2:output:0[model_1/transformer_encoder_4/Encoder-FeedForwardLayer_1_2/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
Dmodel_1/transformer_encoder_4/Encoder-FeedForwardLayer_1_2/TensordotReshapeUmodel_1/transformer_encoder_4/Encoder-FeedForwardLayer_1_2/Tensordot/MatMul:product:0Vmodel_1/transformer_encoder_4/Encoder-FeedForwardLayer_1_2/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������P	�
Qmodel_1/transformer_encoder_4/Encoder-FeedForwardLayer_1_2/BiasAdd/ReadVariableOpReadVariableOpZmodel_1_transformer_encoder_4_encoder_feedforwardlayer_1_2_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype0�
Bmodel_1/transformer_encoder_4/Encoder-FeedForwardLayer_1_2/BiasAddBiasAddMmodel_1/transformer_encoder_4/Encoder-FeedForwardLayer_1_2/Tensordot:output:0Ymodel_1/transformer_encoder_4/Encoder-FeedForwardLayer_1_2/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������P	�
Emodel_1/transformer_encoder_4/Encoder-FeedForwardLayer_1_2/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
Cmodel_1/transformer_encoder_4/Encoder-FeedForwardLayer_1_2/Gelu/mulMulNmodel_1/transformer_encoder_4/Encoder-FeedForwardLayer_1_2/Gelu/mul/x:output:0Kmodel_1/transformer_encoder_4/Encoder-FeedForwardLayer_1_2/BiasAdd:output:0*
T0*+
_output_shapes
:���������P	�
Fmodel_1/transformer_encoder_4/Encoder-FeedForwardLayer_1_2/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *��?�
Gmodel_1/transformer_encoder_4/Encoder-FeedForwardLayer_1_2/Gelu/truedivRealDivKmodel_1/transformer_encoder_4/Encoder-FeedForwardLayer_1_2/BiasAdd:output:0Omodel_1/transformer_encoder_4/Encoder-FeedForwardLayer_1_2/Gelu/Cast/x:output:0*
T0*+
_output_shapes
:���������P	�
Cmodel_1/transformer_encoder_4/Encoder-FeedForwardLayer_1_2/Gelu/ErfErfKmodel_1/transformer_encoder_4/Encoder-FeedForwardLayer_1_2/Gelu/truediv:z:0*
T0*+
_output_shapes
:���������P	�
Emodel_1/transformer_encoder_4/Encoder-FeedForwardLayer_1_2/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
Cmodel_1/transformer_encoder_4/Encoder-FeedForwardLayer_1_2/Gelu/addAddV2Nmodel_1/transformer_encoder_4/Encoder-FeedForwardLayer_1_2/Gelu/add/x:output:0Gmodel_1/transformer_encoder_4/Encoder-FeedForwardLayer_1_2/Gelu/Erf:y:0*
T0*+
_output_shapes
:���������P	�
Emodel_1/transformer_encoder_4/Encoder-FeedForwardLayer_1_2/Gelu/mul_1MulGmodel_1/transformer_encoder_4/Encoder-FeedForwardLayer_1_2/Gelu/mul:z:0Gmodel_1/transformer_encoder_4/Encoder-FeedForwardLayer_1_2/Gelu/add:z:0*
T0*+
_output_shapes
:���������P	�
Smodel_1/transformer_encoder_4/Encoder-FeedForwardLayer_2_2/Tensordot/ReadVariableOpReadVariableOp\model_1_transformer_encoder_4_encoder_feedforwardlayer_2_2_tensordot_readvariableop_resource*
_output_shapes

:		*
dtype0�
Imodel_1/transformer_encoder_4/Encoder-FeedForwardLayer_2_2/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:�
Imodel_1/transformer_encoder_4/Encoder-FeedForwardLayer_2_2/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       �
Jmodel_1/transformer_encoder_4/Encoder-FeedForwardLayer_2_2/Tensordot/ShapeShapeImodel_1/transformer_encoder_4/Encoder-FeedForwardLayer_1_2/Gelu/mul_1:z:0*
T0*
_output_shapes
::���
Rmodel_1/transformer_encoder_4/Encoder-FeedForwardLayer_2_2/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Mmodel_1/transformer_encoder_4/Encoder-FeedForwardLayer_2_2/Tensordot/GatherV2GatherV2Smodel_1/transformer_encoder_4/Encoder-FeedForwardLayer_2_2/Tensordot/Shape:output:0Rmodel_1/transformer_encoder_4/Encoder-FeedForwardLayer_2_2/Tensordot/free:output:0[model_1/transformer_encoder_4/Encoder-FeedForwardLayer_2_2/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
Tmodel_1/transformer_encoder_4/Encoder-FeedForwardLayer_2_2/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Omodel_1/transformer_encoder_4/Encoder-FeedForwardLayer_2_2/Tensordot/GatherV2_1GatherV2Smodel_1/transformer_encoder_4/Encoder-FeedForwardLayer_2_2/Tensordot/Shape:output:0Rmodel_1/transformer_encoder_4/Encoder-FeedForwardLayer_2_2/Tensordot/axes:output:0]model_1/transformer_encoder_4/Encoder-FeedForwardLayer_2_2/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
Jmodel_1/transformer_encoder_4/Encoder-FeedForwardLayer_2_2/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
Imodel_1/transformer_encoder_4/Encoder-FeedForwardLayer_2_2/Tensordot/ProdProdVmodel_1/transformer_encoder_4/Encoder-FeedForwardLayer_2_2/Tensordot/GatherV2:output:0Smodel_1/transformer_encoder_4/Encoder-FeedForwardLayer_2_2/Tensordot/Const:output:0*
T0*
_output_shapes
: �
Lmodel_1/transformer_encoder_4/Encoder-FeedForwardLayer_2_2/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
Kmodel_1/transformer_encoder_4/Encoder-FeedForwardLayer_2_2/Tensordot/Prod_1ProdXmodel_1/transformer_encoder_4/Encoder-FeedForwardLayer_2_2/Tensordot/GatherV2_1:output:0Umodel_1/transformer_encoder_4/Encoder-FeedForwardLayer_2_2/Tensordot/Const_1:output:0*
T0*
_output_shapes
: �
Pmodel_1/transformer_encoder_4/Encoder-FeedForwardLayer_2_2/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Kmodel_1/transformer_encoder_4/Encoder-FeedForwardLayer_2_2/Tensordot/concatConcatV2Rmodel_1/transformer_encoder_4/Encoder-FeedForwardLayer_2_2/Tensordot/free:output:0Rmodel_1/transformer_encoder_4/Encoder-FeedForwardLayer_2_2/Tensordot/axes:output:0Ymodel_1/transformer_encoder_4/Encoder-FeedForwardLayer_2_2/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
Jmodel_1/transformer_encoder_4/Encoder-FeedForwardLayer_2_2/Tensordot/stackPackRmodel_1/transformer_encoder_4/Encoder-FeedForwardLayer_2_2/Tensordot/Prod:output:0Tmodel_1/transformer_encoder_4/Encoder-FeedForwardLayer_2_2/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
Nmodel_1/transformer_encoder_4/Encoder-FeedForwardLayer_2_2/Tensordot/transpose	TransposeImodel_1/transformer_encoder_4/Encoder-FeedForwardLayer_1_2/Gelu/mul_1:z:0Tmodel_1/transformer_encoder_4/Encoder-FeedForwardLayer_2_2/Tensordot/concat:output:0*
T0*+
_output_shapes
:���������P	�
Lmodel_1/transformer_encoder_4/Encoder-FeedForwardLayer_2_2/Tensordot/ReshapeReshapeRmodel_1/transformer_encoder_4/Encoder-FeedForwardLayer_2_2/Tensordot/transpose:y:0Smodel_1/transformer_encoder_4/Encoder-FeedForwardLayer_2_2/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
Kmodel_1/transformer_encoder_4/Encoder-FeedForwardLayer_2_2/Tensordot/MatMulMatMulUmodel_1/transformer_encoder_4/Encoder-FeedForwardLayer_2_2/Tensordot/Reshape:output:0[model_1/transformer_encoder_4/Encoder-FeedForwardLayer_2_2/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������	�
Lmodel_1/transformer_encoder_4/Encoder-FeedForwardLayer_2_2/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:	�
Rmodel_1/transformer_encoder_4/Encoder-FeedForwardLayer_2_2/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Mmodel_1/transformer_encoder_4/Encoder-FeedForwardLayer_2_2/Tensordot/concat_1ConcatV2Vmodel_1/transformer_encoder_4/Encoder-FeedForwardLayer_2_2/Tensordot/GatherV2:output:0Umodel_1/transformer_encoder_4/Encoder-FeedForwardLayer_2_2/Tensordot/Const_2:output:0[model_1/transformer_encoder_4/Encoder-FeedForwardLayer_2_2/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
Dmodel_1/transformer_encoder_4/Encoder-FeedForwardLayer_2_2/TensordotReshapeUmodel_1/transformer_encoder_4/Encoder-FeedForwardLayer_2_2/Tensordot/MatMul:product:0Vmodel_1/transformer_encoder_4/Encoder-FeedForwardLayer_2_2/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������P	�
Qmodel_1/transformer_encoder_4/Encoder-FeedForwardLayer_2_2/BiasAdd/ReadVariableOpReadVariableOpZmodel_1_transformer_encoder_4_encoder_feedforwardlayer_2_2_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype0�
Bmodel_1/transformer_encoder_4/Encoder-FeedForwardLayer_2_2/BiasAddBiasAddMmodel_1/transformer_encoder_4/Encoder-FeedForwardLayer_2_2/Tensordot:output:0Ymodel_1/transformer_encoder_4/Encoder-FeedForwardLayer_2_2/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������P	�
0model_1/transformer_encoder_4/dropout_4/IdentityIdentityKmodel_1/transformer_encoder_4/Encoder-FeedForwardLayer_2_2/BiasAdd:output:0*
T0*+
_output_shapes
:���������P	�
=model_1/transformer_encoder_4/Encoder-2nd-AdditionLayer-2/addAddV2Amodel_1/transformer_encoder_4/Encoder-1st-AdditionLayer-2/add:z:09model_1/transformer_encoder_4/dropout_4/Identity:output:0*
T0*+
_output_shapes
:���������P	�
Dmodel_1/transformer_encoder_5/Encoder-1st-NormalizationLayer-3/ShapeShapeAmodel_1/transformer_encoder_4/Encoder-2nd-AdditionLayer-2/add:z:0*
T0*
_output_shapes
::���
Rmodel_1/transformer_encoder_5/Encoder-1st-NormalizationLayer-3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
Tmodel_1/transformer_encoder_5/Encoder-1st-NormalizationLayer-3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
Tmodel_1/transformer_encoder_5/Encoder-1st-NormalizationLayer-3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
Lmodel_1/transformer_encoder_5/Encoder-1st-NormalizationLayer-3/strided_sliceStridedSliceMmodel_1/transformer_encoder_5/Encoder-1st-NormalizationLayer-3/Shape:output:0[model_1/transformer_encoder_5/Encoder-1st-NormalizationLayer-3/strided_slice/stack:output:0]model_1/transformer_encoder_5/Encoder-1st-NormalizationLayer-3/strided_slice/stack_1:output:0]model_1/transformer_encoder_5/Encoder-1st-NormalizationLayer-3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask�
Dmodel_1/transformer_encoder_5/Encoder-1st-NormalizationLayer-3/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
Cmodel_1/transformer_encoder_5/Encoder-1st-NormalizationLayer-3/ProdProdUmodel_1/transformer_encoder_5/Encoder-1st-NormalizationLayer-3/strided_slice:output:0Mmodel_1/transformer_encoder_5/Encoder-1st-NormalizationLayer-3/Const:output:0*
T0*
_output_shapes
: �
Tmodel_1/transformer_encoder_5/Encoder-1st-NormalizationLayer-3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:�
Vmodel_1/transformer_encoder_5/Encoder-1st-NormalizationLayer-3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: �
Vmodel_1/transformer_encoder_5/Encoder-1st-NormalizationLayer-3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
Nmodel_1/transformer_encoder_5/Encoder-1st-NormalizationLayer-3/strided_slice_1StridedSliceMmodel_1/transformer_encoder_5/Encoder-1st-NormalizationLayer-3/Shape:output:0]model_1/transformer_encoder_5/Encoder-1st-NormalizationLayer-3/strided_slice_1/stack:output:0_model_1/transformer_encoder_5/Encoder-1st-NormalizationLayer-3/strided_slice_1/stack_1:output:0_model_1/transformer_encoder_5/Encoder-1st-NormalizationLayer-3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask�
Fmodel_1/transformer_encoder_5/Encoder-1st-NormalizationLayer-3/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
Emodel_1/transformer_encoder_5/Encoder-1st-NormalizationLayer-3/Prod_1ProdWmodel_1/transformer_encoder_5/Encoder-1st-NormalizationLayer-3/strided_slice_1:output:0Omodel_1/transformer_encoder_5/Encoder-1st-NormalizationLayer-3/Const_1:output:0*
T0*
_output_shapes
: �
Nmodel_1/transformer_encoder_5/Encoder-1st-NormalizationLayer-3/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :�
Nmodel_1/transformer_encoder_5/Encoder-1st-NormalizationLayer-3/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :�
Lmodel_1/transformer_encoder_5/Encoder-1st-NormalizationLayer-3/Reshape/shapePackWmodel_1/transformer_encoder_5/Encoder-1st-NormalizationLayer-3/Reshape/shape/0:output:0Lmodel_1/transformer_encoder_5/Encoder-1st-NormalizationLayer-3/Prod:output:0Nmodel_1/transformer_encoder_5/Encoder-1st-NormalizationLayer-3/Prod_1:output:0Wmodel_1/transformer_encoder_5/Encoder-1st-NormalizationLayer-3/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:�
Fmodel_1/transformer_encoder_5/Encoder-1st-NormalizationLayer-3/ReshapeReshapeAmodel_1/transformer_encoder_4/Encoder-2nd-AdditionLayer-2/add:z:0Umodel_1/transformer_encoder_5/Encoder-1st-NormalizationLayer-3/Reshape/shape:output:0*
T0*8
_output_shapes&
$:"�������������������
Jmodel_1/transformer_encoder_5/Encoder-1st-NormalizationLayer-3/ones/packedPackLmodel_1/transformer_encoder_5/Encoder-1st-NormalizationLayer-3/Prod:output:0*
N*
T0*
_output_shapes
:�
Imodel_1/transformer_encoder_5/Encoder-1st-NormalizationLayer-3/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
Cmodel_1/transformer_encoder_5/Encoder-1st-NormalizationLayer-3/onesFillSmodel_1/transformer_encoder_5/Encoder-1st-NormalizationLayer-3/ones/packed:output:0Rmodel_1/transformer_encoder_5/Encoder-1st-NormalizationLayer-3/ones/Const:output:0*
T0*#
_output_shapes
:����������
Kmodel_1/transformer_encoder_5/Encoder-1st-NormalizationLayer-3/zeros/packedPackLmodel_1/transformer_encoder_5/Encoder-1st-NormalizationLayer-3/Prod:output:0*
N*
T0*
_output_shapes
:�
Jmodel_1/transformer_encoder_5/Encoder-1st-NormalizationLayer-3/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
Dmodel_1/transformer_encoder_5/Encoder-1st-NormalizationLayer-3/zerosFillTmodel_1/transformer_encoder_5/Encoder-1st-NormalizationLayer-3/zeros/packed:output:0Smodel_1/transformer_encoder_5/Encoder-1st-NormalizationLayer-3/zeros/Const:output:0*
T0*#
_output_shapes
:����������
Fmodel_1/transformer_encoder_5/Encoder-1st-NormalizationLayer-3/Const_2Const*
_output_shapes
: *
dtype0*
valueB �
Fmodel_1/transformer_encoder_5/Encoder-1st-NormalizationLayer-3/Const_3Const*
_output_shapes
: *
dtype0*
valueB �
Omodel_1/transformer_encoder_5/Encoder-1st-NormalizationLayer-3/FusedBatchNormV3FusedBatchNormV3Omodel_1/transformer_encoder_5/Encoder-1st-NormalizationLayer-3/Reshape:output:0Lmodel_1/transformer_encoder_5/Encoder-1st-NormalizationLayer-3/ones:output:0Mmodel_1/transformer_encoder_5/Encoder-1st-NormalizationLayer-3/zeros:output:0Omodel_1/transformer_encoder_5/Encoder-1st-NormalizationLayer-3/Const_2:output:0Omodel_1/transformer_encoder_5/Encoder-1st-NormalizationLayer-3/Const_3:output:0*
T0*
U0*x
_output_shapesf
d:"������������������:���������:���������:���������:���������:*
data_formatNCHW*
epsilon%o�:�
Hmodel_1/transformer_encoder_5/Encoder-1st-NormalizationLayer-3/Reshape_1ReshapeSmodel_1/transformer_encoder_5/Encoder-1st-NormalizationLayer-3/FusedBatchNormV3:y:0Mmodel_1/transformer_encoder_5/Encoder-1st-NormalizationLayer-3/Shape:output:0*
T0*+
_output_shapes
:���������P	�
Qmodel_1/transformer_encoder_5/Encoder-1st-NormalizationLayer-3/mul/ReadVariableOpReadVariableOpZmodel_1_transformer_encoder_5_encoder_1st_normalizationlayer_3_mul_readvariableop_resource*
_output_shapes
:	*
dtype0�
Bmodel_1/transformer_encoder_5/Encoder-1st-NormalizationLayer-3/mulMulQmodel_1/transformer_encoder_5/Encoder-1st-NormalizationLayer-3/Reshape_1:output:0Ymodel_1/transformer_encoder_5/Encoder-1st-NormalizationLayer-3/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������P	�
Qmodel_1/transformer_encoder_5/Encoder-1st-NormalizationLayer-3/add/ReadVariableOpReadVariableOpZmodel_1_transformer_encoder_5_encoder_1st_normalizationlayer_3_add_readvariableop_resource*
_output_shapes
:	*
dtype0�
Bmodel_1/transformer_encoder_5/Encoder-1st-NormalizationLayer-3/addAddV2Fmodel_1/transformer_encoder_5/Encoder-1st-NormalizationLayer-3/mul:z:0Ymodel_1/transformer_encoder_5/Encoder-1st-NormalizationLayer-3/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������P	�
]model_1/transformer_encoder_5/Encoder-SelfAttentionLayer-3/query/einsum/Einsum/ReadVariableOpReadVariableOpfmodel_1_transformer_encoder_5_encoder_selfattentionlayer_3_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:	*
dtype0�
Nmodel_1/transformer_encoder_5/Encoder-SelfAttentionLayer-3/query/einsum/EinsumEinsumFmodel_1/transformer_encoder_5/Encoder-1st-NormalizationLayer-3/add:z:0emodel_1/transformer_encoder_5/Encoder-SelfAttentionLayer-3/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������P*
equationabc,cde->abde�
Smodel_1/transformer_encoder_5/Encoder-SelfAttentionLayer-3/query/add/ReadVariableOpReadVariableOp\model_1_transformer_encoder_5_encoder_selfattentionlayer_3_query_add_readvariableop_resource*
_output_shapes

:*
dtype0�
Dmodel_1/transformer_encoder_5/Encoder-SelfAttentionLayer-3/query/addAddV2Wmodel_1/transformer_encoder_5/Encoder-SelfAttentionLayer-3/query/einsum/Einsum:output:0[model_1/transformer_encoder_5/Encoder-SelfAttentionLayer-3/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������P�
[model_1/transformer_encoder_5/Encoder-SelfAttentionLayer-3/key/einsum/Einsum/ReadVariableOpReadVariableOpdmodel_1_transformer_encoder_5_encoder_selfattentionlayer_3_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:	*
dtype0�
Lmodel_1/transformer_encoder_5/Encoder-SelfAttentionLayer-3/key/einsum/EinsumEinsumFmodel_1/transformer_encoder_5/Encoder-1st-NormalizationLayer-3/add:z:0cmodel_1/transformer_encoder_5/Encoder-SelfAttentionLayer-3/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������P*
equationabc,cde->abde�
Qmodel_1/transformer_encoder_5/Encoder-SelfAttentionLayer-3/key/add/ReadVariableOpReadVariableOpZmodel_1_transformer_encoder_5_encoder_selfattentionlayer_3_key_add_readvariableop_resource*
_output_shapes

:*
dtype0�
Bmodel_1/transformer_encoder_5/Encoder-SelfAttentionLayer-3/key/addAddV2Umodel_1/transformer_encoder_5/Encoder-SelfAttentionLayer-3/key/einsum/Einsum:output:0Ymodel_1/transformer_encoder_5/Encoder-SelfAttentionLayer-3/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������P�
]model_1/transformer_encoder_5/Encoder-SelfAttentionLayer-3/value/einsum/Einsum/ReadVariableOpReadVariableOpfmodel_1_transformer_encoder_5_encoder_selfattentionlayer_3_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:	*
dtype0�
Nmodel_1/transformer_encoder_5/Encoder-SelfAttentionLayer-3/value/einsum/EinsumEinsumFmodel_1/transformer_encoder_5/Encoder-1st-NormalizationLayer-3/add:z:0emodel_1/transformer_encoder_5/Encoder-SelfAttentionLayer-3/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������P*
equationabc,cde->abde�
Smodel_1/transformer_encoder_5/Encoder-SelfAttentionLayer-3/value/add/ReadVariableOpReadVariableOp\model_1_transformer_encoder_5_encoder_selfattentionlayer_3_value_add_readvariableop_resource*
_output_shapes

:*
dtype0�
Dmodel_1/transformer_encoder_5/Encoder-SelfAttentionLayer-3/value/addAddV2Wmodel_1/transformer_encoder_5/Encoder-SelfAttentionLayer-3/value/einsum/Einsum:output:0[model_1/transformer_encoder_5/Encoder-SelfAttentionLayer-3/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������P�
@model_1/transformer_encoder_5/Encoder-SelfAttentionLayer-3/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *:�?�
>model_1/transformer_encoder_5/Encoder-SelfAttentionLayer-3/MulMulHmodel_1/transformer_encoder_5/Encoder-SelfAttentionLayer-3/query/add:z:0Imodel_1/transformer_encoder_5/Encoder-SelfAttentionLayer-3/Mul/y:output:0*
T0*/
_output_shapes
:���������P�
Hmodel_1/transformer_encoder_5/Encoder-SelfAttentionLayer-3/einsum/EinsumEinsumFmodel_1/transformer_encoder_5/Encoder-SelfAttentionLayer-3/key/add:z:0Bmodel_1/transformer_encoder_5/Encoder-SelfAttentionLayer-3/Mul:z:0*
N*
T0*/
_output_shapes
:���������PP*
equationaecd,abcd->acbe�
Jmodel_1/transformer_encoder_5/Encoder-SelfAttentionLayer-3/softmax/SoftmaxSoftmaxQmodel_1/transformer_encoder_5/Encoder-SelfAttentionLayer-3/einsum/Einsum:output:0*
T0*/
_output_shapes
:���������PP�
Kmodel_1/transformer_encoder_5/Encoder-SelfAttentionLayer-3/dropout/IdentityIdentityTmodel_1/transformer_encoder_5/Encoder-SelfAttentionLayer-3/softmax/Softmax:softmax:0*
T0*/
_output_shapes
:���������PP�
Jmodel_1/transformer_encoder_5/Encoder-SelfAttentionLayer-3/einsum_1/EinsumEinsumTmodel_1/transformer_encoder_5/Encoder-SelfAttentionLayer-3/dropout/Identity:output:0Hmodel_1/transformer_encoder_5/Encoder-SelfAttentionLayer-3/value/add:z:0*
N*
T0*/
_output_shapes
:���������P*
equationacbe,aecd->abcd�
hmodel_1/transformer_encoder_5/Encoder-SelfAttentionLayer-3/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpqmodel_1_transformer_encoder_5_encoder_selfattentionlayer_3_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:	*
dtype0�
Ymodel_1/transformer_encoder_5/Encoder-SelfAttentionLayer-3/attention_output/einsum/EinsumEinsumSmodel_1/transformer_encoder_5/Encoder-SelfAttentionLayer-3/einsum_1/Einsum:output:0pmodel_1/transformer_encoder_5/Encoder-SelfAttentionLayer-3/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:���������P	*
equationabcd,cde->abe�
^model_1/transformer_encoder_5/Encoder-SelfAttentionLayer-3/attention_output/add/ReadVariableOpReadVariableOpgmodel_1_transformer_encoder_5_encoder_selfattentionlayer_3_attention_output_add_readvariableop_resource*
_output_shapes
:	*
dtype0�
Omodel_1/transformer_encoder_5/Encoder-SelfAttentionLayer-3/attention_output/addAddV2bmodel_1/transformer_encoder_5/Encoder-SelfAttentionLayer-3/attention_output/einsum/Einsum:output:0fmodel_1/transformer_encoder_5/Encoder-SelfAttentionLayer-3/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������P	�
=model_1/transformer_encoder_5/Encoder-1st-AdditionLayer-3/addAddV2Amodel_1/transformer_encoder_4/Encoder-2nd-AdditionLayer-2/add:z:0Smodel_1/transformer_encoder_5/Encoder-SelfAttentionLayer-3/attention_output/add:z:0*
T0*+
_output_shapes
:���������P	�
Dmodel_1/transformer_encoder_5/Encoder-2nd-NormalizationLayer-3/ShapeShapeAmodel_1/transformer_encoder_5/Encoder-1st-AdditionLayer-3/add:z:0*
T0*
_output_shapes
::���
Rmodel_1/transformer_encoder_5/Encoder-2nd-NormalizationLayer-3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
Tmodel_1/transformer_encoder_5/Encoder-2nd-NormalizationLayer-3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
Tmodel_1/transformer_encoder_5/Encoder-2nd-NormalizationLayer-3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
Lmodel_1/transformer_encoder_5/Encoder-2nd-NormalizationLayer-3/strided_sliceStridedSliceMmodel_1/transformer_encoder_5/Encoder-2nd-NormalizationLayer-3/Shape:output:0[model_1/transformer_encoder_5/Encoder-2nd-NormalizationLayer-3/strided_slice/stack:output:0]model_1/transformer_encoder_5/Encoder-2nd-NormalizationLayer-3/strided_slice/stack_1:output:0]model_1/transformer_encoder_5/Encoder-2nd-NormalizationLayer-3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask�
Dmodel_1/transformer_encoder_5/Encoder-2nd-NormalizationLayer-3/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
Cmodel_1/transformer_encoder_5/Encoder-2nd-NormalizationLayer-3/ProdProdUmodel_1/transformer_encoder_5/Encoder-2nd-NormalizationLayer-3/strided_slice:output:0Mmodel_1/transformer_encoder_5/Encoder-2nd-NormalizationLayer-3/Const:output:0*
T0*
_output_shapes
: �
Tmodel_1/transformer_encoder_5/Encoder-2nd-NormalizationLayer-3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:�
Vmodel_1/transformer_encoder_5/Encoder-2nd-NormalizationLayer-3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: �
Vmodel_1/transformer_encoder_5/Encoder-2nd-NormalizationLayer-3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
Nmodel_1/transformer_encoder_5/Encoder-2nd-NormalizationLayer-3/strided_slice_1StridedSliceMmodel_1/transformer_encoder_5/Encoder-2nd-NormalizationLayer-3/Shape:output:0]model_1/transformer_encoder_5/Encoder-2nd-NormalizationLayer-3/strided_slice_1/stack:output:0_model_1/transformer_encoder_5/Encoder-2nd-NormalizationLayer-3/strided_slice_1/stack_1:output:0_model_1/transformer_encoder_5/Encoder-2nd-NormalizationLayer-3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask�
Fmodel_1/transformer_encoder_5/Encoder-2nd-NormalizationLayer-3/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
Emodel_1/transformer_encoder_5/Encoder-2nd-NormalizationLayer-3/Prod_1ProdWmodel_1/transformer_encoder_5/Encoder-2nd-NormalizationLayer-3/strided_slice_1:output:0Omodel_1/transformer_encoder_5/Encoder-2nd-NormalizationLayer-3/Const_1:output:0*
T0*
_output_shapes
: �
Nmodel_1/transformer_encoder_5/Encoder-2nd-NormalizationLayer-3/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :�
Nmodel_1/transformer_encoder_5/Encoder-2nd-NormalizationLayer-3/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :�
Lmodel_1/transformer_encoder_5/Encoder-2nd-NormalizationLayer-3/Reshape/shapePackWmodel_1/transformer_encoder_5/Encoder-2nd-NormalizationLayer-3/Reshape/shape/0:output:0Lmodel_1/transformer_encoder_5/Encoder-2nd-NormalizationLayer-3/Prod:output:0Nmodel_1/transformer_encoder_5/Encoder-2nd-NormalizationLayer-3/Prod_1:output:0Wmodel_1/transformer_encoder_5/Encoder-2nd-NormalizationLayer-3/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:�
Fmodel_1/transformer_encoder_5/Encoder-2nd-NormalizationLayer-3/ReshapeReshapeAmodel_1/transformer_encoder_5/Encoder-1st-AdditionLayer-3/add:z:0Umodel_1/transformer_encoder_5/Encoder-2nd-NormalizationLayer-3/Reshape/shape:output:0*
T0*8
_output_shapes&
$:"�������������������
Jmodel_1/transformer_encoder_5/Encoder-2nd-NormalizationLayer-3/ones/packedPackLmodel_1/transformer_encoder_5/Encoder-2nd-NormalizationLayer-3/Prod:output:0*
N*
T0*
_output_shapes
:�
Imodel_1/transformer_encoder_5/Encoder-2nd-NormalizationLayer-3/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
Cmodel_1/transformer_encoder_5/Encoder-2nd-NormalizationLayer-3/onesFillSmodel_1/transformer_encoder_5/Encoder-2nd-NormalizationLayer-3/ones/packed:output:0Rmodel_1/transformer_encoder_5/Encoder-2nd-NormalizationLayer-3/ones/Const:output:0*
T0*#
_output_shapes
:����������
Kmodel_1/transformer_encoder_5/Encoder-2nd-NormalizationLayer-3/zeros/packedPackLmodel_1/transformer_encoder_5/Encoder-2nd-NormalizationLayer-3/Prod:output:0*
N*
T0*
_output_shapes
:�
Jmodel_1/transformer_encoder_5/Encoder-2nd-NormalizationLayer-3/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
Dmodel_1/transformer_encoder_5/Encoder-2nd-NormalizationLayer-3/zerosFillTmodel_1/transformer_encoder_5/Encoder-2nd-NormalizationLayer-3/zeros/packed:output:0Smodel_1/transformer_encoder_5/Encoder-2nd-NormalizationLayer-3/zeros/Const:output:0*
T0*#
_output_shapes
:����������
Fmodel_1/transformer_encoder_5/Encoder-2nd-NormalizationLayer-3/Const_2Const*
_output_shapes
: *
dtype0*
valueB �
Fmodel_1/transformer_encoder_5/Encoder-2nd-NormalizationLayer-3/Const_3Const*
_output_shapes
: *
dtype0*
valueB �
Omodel_1/transformer_encoder_5/Encoder-2nd-NormalizationLayer-3/FusedBatchNormV3FusedBatchNormV3Omodel_1/transformer_encoder_5/Encoder-2nd-NormalizationLayer-3/Reshape:output:0Lmodel_1/transformer_encoder_5/Encoder-2nd-NormalizationLayer-3/ones:output:0Mmodel_1/transformer_encoder_5/Encoder-2nd-NormalizationLayer-3/zeros:output:0Omodel_1/transformer_encoder_5/Encoder-2nd-NormalizationLayer-3/Const_2:output:0Omodel_1/transformer_encoder_5/Encoder-2nd-NormalizationLayer-3/Const_3:output:0*
T0*
U0*x
_output_shapesf
d:"������������������:���������:���������:���������:���������:*
data_formatNCHW*
epsilon%o�:�
Hmodel_1/transformer_encoder_5/Encoder-2nd-NormalizationLayer-3/Reshape_1ReshapeSmodel_1/transformer_encoder_5/Encoder-2nd-NormalizationLayer-3/FusedBatchNormV3:y:0Mmodel_1/transformer_encoder_5/Encoder-2nd-NormalizationLayer-3/Shape:output:0*
T0*+
_output_shapes
:���������P	�
Qmodel_1/transformer_encoder_5/Encoder-2nd-NormalizationLayer-3/mul/ReadVariableOpReadVariableOpZmodel_1_transformer_encoder_5_encoder_2nd_normalizationlayer_3_mul_readvariableop_resource*
_output_shapes
:	*
dtype0�
Bmodel_1/transformer_encoder_5/Encoder-2nd-NormalizationLayer-3/mulMulQmodel_1/transformer_encoder_5/Encoder-2nd-NormalizationLayer-3/Reshape_1:output:0Ymodel_1/transformer_encoder_5/Encoder-2nd-NormalizationLayer-3/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������P	�
Qmodel_1/transformer_encoder_5/Encoder-2nd-NormalizationLayer-3/add/ReadVariableOpReadVariableOpZmodel_1_transformer_encoder_5_encoder_2nd_normalizationlayer_3_add_readvariableop_resource*
_output_shapes
:	*
dtype0�
Bmodel_1/transformer_encoder_5/Encoder-2nd-NormalizationLayer-3/addAddV2Fmodel_1/transformer_encoder_5/Encoder-2nd-NormalizationLayer-3/mul:z:0Ymodel_1/transformer_encoder_5/Encoder-2nd-NormalizationLayer-3/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������P	�
Smodel_1/transformer_encoder_5/Encoder-FeedForwardLayer_1_3/Tensordot/ReadVariableOpReadVariableOp\model_1_transformer_encoder_5_encoder_feedforwardlayer_1_3_tensordot_readvariableop_resource*
_output_shapes

:		*
dtype0�
Imodel_1/transformer_encoder_5/Encoder-FeedForwardLayer_1_3/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:�
Imodel_1/transformer_encoder_5/Encoder-FeedForwardLayer_1_3/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       �
Jmodel_1/transformer_encoder_5/Encoder-FeedForwardLayer_1_3/Tensordot/ShapeShapeFmodel_1/transformer_encoder_5/Encoder-2nd-NormalizationLayer-3/add:z:0*
T0*
_output_shapes
::���
Rmodel_1/transformer_encoder_5/Encoder-FeedForwardLayer_1_3/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Mmodel_1/transformer_encoder_5/Encoder-FeedForwardLayer_1_3/Tensordot/GatherV2GatherV2Smodel_1/transformer_encoder_5/Encoder-FeedForwardLayer_1_3/Tensordot/Shape:output:0Rmodel_1/transformer_encoder_5/Encoder-FeedForwardLayer_1_3/Tensordot/free:output:0[model_1/transformer_encoder_5/Encoder-FeedForwardLayer_1_3/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
Tmodel_1/transformer_encoder_5/Encoder-FeedForwardLayer_1_3/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Omodel_1/transformer_encoder_5/Encoder-FeedForwardLayer_1_3/Tensordot/GatherV2_1GatherV2Smodel_1/transformer_encoder_5/Encoder-FeedForwardLayer_1_3/Tensordot/Shape:output:0Rmodel_1/transformer_encoder_5/Encoder-FeedForwardLayer_1_3/Tensordot/axes:output:0]model_1/transformer_encoder_5/Encoder-FeedForwardLayer_1_3/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
Jmodel_1/transformer_encoder_5/Encoder-FeedForwardLayer_1_3/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
Imodel_1/transformer_encoder_5/Encoder-FeedForwardLayer_1_3/Tensordot/ProdProdVmodel_1/transformer_encoder_5/Encoder-FeedForwardLayer_1_3/Tensordot/GatherV2:output:0Smodel_1/transformer_encoder_5/Encoder-FeedForwardLayer_1_3/Tensordot/Const:output:0*
T0*
_output_shapes
: �
Lmodel_1/transformer_encoder_5/Encoder-FeedForwardLayer_1_3/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
Kmodel_1/transformer_encoder_5/Encoder-FeedForwardLayer_1_3/Tensordot/Prod_1ProdXmodel_1/transformer_encoder_5/Encoder-FeedForwardLayer_1_3/Tensordot/GatherV2_1:output:0Umodel_1/transformer_encoder_5/Encoder-FeedForwardLayer_1_3/Tensordot/Const_1:output:0*
T0*
_output_shapes
: �
Pmodel_1/transformer_encoder_5/Encoder-FeedForwardLayer_1_3/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Kmodel_1/transformer_encoder_5/Encoder-FeedForwardLayer_1_3/Tensordot/concatConcatV2Rmodel_1/transformer_encoder_5/Encoder-FeedForwardLayer_1_3/Tensordot/free:output:0Rmodel_1/transformer_encoder_5/Encoder-FeedForwardLayer_1_3/Tensordot/axes:output:0Ymodel_1/transformer_encoder_5/Encoder-FeedForwardLayer_1_3/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
Jmodel_1/transformer_encoder_5/Encoder-FeedForwardLayer_1_3/Tensordot/stackPackRmodel_1/transformer_encoder_5/Encoder-FeedForwardLayer_1_3/Tensordot/Prod:output:0Tmodel_1/transformer_encoder_5/Encoder-FeedForwardLayer_1_3/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
Nmodel_1/transformer_encoder_5/Encoder-FeedForwardLayer_1_3/Tensordot/transpose	TransposeFmodel_1/transformer_encoder_5/Encoder-2nd-NormalizationLayer-3/add:z:0Tmodel_1/transformer_encoder_5/Encoder-FeedForwardLayer_1_3/Tensordot/concat:output:0*
T0*+
_output_shapes
:���������P	�
Lmodel_1/transformer_encoder_5/Encoder-FeedForwardLayer_1_3/Tensordot/ReshapeReshapeRmodel_1/transformer_encoder_5/Encoder-FeedForwardLayer_1_3/Tensordot/transpose:y:0Smodel_1/transformer_encoder_5/Encoder-FeedForwardLayer_1_3/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
Kmodel_1/transformer_encoder_5/Encoder-FeedForwardLayer_1_3/Tensordot/MatMulMatMulUmodel_1/transformer_encoder_5/Encoder-FeedForwardLayer_1_3/Tensordot/Reshape:output:0[model_1/transformer_encoder_5/Encoder-FeedForwardLayer_1_3/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������	�
Lmodel_1/transformer_encoder_5/Encoder-FeedForwardLayer_1_3/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:	�
Rmodel_1/transformer_encoder_5/Encoder-FeedForwardLayer_1_3/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Mmodel_1/transformer_encoder_5/Encoder-FeedForwardLayer_1_3/Tensordot/concat_1ConcatV2Vmodel_1/transformer_encoder_5/Encoder-FeedForwardLayer_1_3/Tensordot/GatherV2:output:0Umodel_1/transformer_encoder_5/Encoder-FeedForwardLayer_1_3/Tensordot/Const_2:output:0[model_1/transformer_encoder_5/Encoder-FeedForwardLayer_1_3/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
Dmodel_1/transformer_encoder_5/Encoder-FeedForwardLayer_1_3/TensordotReshapeUmodel_1/transformer_encoder_5/Encoder-FeedForwardLayer_1_3/Tensordot/MatMul:product:0Vmodel_1/transformer_encoder_5/Encoder-FeedForwardLayer_1_3/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������P	�
Qmodel_1/transformer_encoder_5/Encoder-FeedForwardLayer_1_3/BiasAdd/ReadVariableOpReadVariableOpZmodel_1_transformer_encoder_5_encoder_feedforwardlayer_1_3_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype0�
Bmodel_1/transformer_encoder_5/Encoder-FeedForwardLayer_1_3/BiasAddBiasAddMmodel_1/transformer_encoder_5/Encoder-FeedForwardLayer_1_3/Tensordot:output:0Ymodel_1/transformer_encoder_5/Encoder-FeedForwardLayer_1_3/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������P	�
Emodel_1/transformer_encoder_5/Encoder-FeedForwardLayer_1_3/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
Cmodel_1/transformer_encoder_5/Encoder-FeedForwardLayer_1_3/Gelu/mulMulNmodel_1/transformer_encoder_5/Encoder-FeedForwardLayer_1_3/Gelu/mul/x:output:0Kmodel_1/transformer_encoder_5/Encoder-FeedForwardLayer_1_3/BiasAdd:output:0*
T0*+
_output_shapes
:���������P	�
Fmodel_1/transformer_encoder_5/Encoder-FeedForwardLayer_1_3/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *��?�
Gmodel_1/transformer_encoder_5/Encoder-FeedForwardLayer_1_3/Gelu/truedivRealDivKmodel_1/transformer_encoder_5/Encoder-FeedForwardLayer_1_3/BiasAdd:output:0Omodel_1/transformer_encoder_5/Encoder-FeedForwardLayer_1_3/Gelu/Cast/x:output:0*
T0*+
_output_shapes
:���������P	�
Cmodel_1/transformer_encoder_5/Encoder-FeedForwardLayer_1_3/Gelu/ErfErfKmodel_1/transformer_encoder_5/Encoder-FeedForwardLayer_1_3/Gelu/truediv:z:0*
T0*+
_output_shapes
:���������P	�
Emodel_1/transformer_encoder_5/Encoder-FeedForwardLayer_1_3/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
Cmodel_1/transformer_encoder_5/Encoder-FeedForwardLayer_1_3/Gelu/addAddV2Nmodel_1/transformer_encoder_5/Encoder-FeedForwardLayer_1_3/Gelu/add/x:output:0Gmodel_1/transformer_encoder_5/Encoder-FeedForwardLayer_1_3/Gelu/Erf:y:0*
T0*+
_output_shapes
:���������P	�
Emodel_1/transformer_encoder_5/Encoder-FeedForwardLayer_1_3/Gelu/mul_1MulGmodel_1/transformer_encoder_5/Encoder-FeedForwardLayer_1_3/Gelu/mul:z:0Gmodel_1/transformer_encoder_5/Encoder-FeedForwardLayer_1_3/Gelu/add:z:0*
T0*+
_output_shapes
:���������P	�
Smodel_1/transformer_encoder_5/Encoder-FeedForwardLayer_2_3/Tensordot/ReadVariableOpReadVariableOp\model_1_transformer_encoder_5_encoder_feedforwardlayer_2_3_tensordot_readvariableop_resource*
_output_shapes

:		*
dtype0�
Imodel_1/transformer_encoder_5/Encoder-FeedForwardLayer_2_3/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:�
Imodel_1/transformer_encoder_5/Encoder-FeedForwardLayer_2_3/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       �
Jmodel_1/transformer_encoder_5/Encoder-FeedForwardLayer_2_3/Tensordot/ShapeShapeImodel_1/transformer_encoder_5/Encoder-FeedForwardLayer_1_3/Gelu/mul_1:z:0*
T0*
_output_shapes
::���
Rmodel_1/transformer_encoder_5/Encoder-FeedForwardLayer_2_3/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Mmodel_1/transformer_encoder_5/Encoder-FeedForwardLayer_2_3/Tensordot/GatherV2GatherV2Smodel_1/transformer_encoder_5/Encoder-FeedForwardLayer_2_3/Tensordot/Shape:output:0Rmodel_1/transformer_encoder_5/Encoder-FeedForwardLayer_2_3/Tensordot/free:output:0[model_1/transformer_encoder_5/Encoder-FeedForwardLayer_2_3/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
Tmodel_1/transformer_encoder_5/Encoder-FeedForwardLayer_2_3/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Omodel_1/transformer_encoder_5/Encoder-FeedForwardLayer_2_3/Tensordot/GatherV2_1GatherV2Smodel_1/transformer_encoder_5/Encoder-FeedForwardLayer_2_3/Tensordot/Shape:output:0Rmodel_1/transformer_encoder_5/Encoder-FeedForwardLayer_2_3/Tensordot/axes:output:0]model_1/transformer_encoder_5/Encoder-FeedForwardLayer_2_3/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
Jmodel_1/transformer_encoder_5/Encoder-FeedForwardLayer_2_3/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
Imodel_1/transformer_encoder_5/Encoder-FeedForwardLayer_2_3/Tensordot/ProdProdVmodel_1/transformer_encoder_5/Encoder-FeedForwardLayer_2_3/Tensordot/GatherV2:output:0Smodel_1/transformer_encoder_5/Encoder-FeedForwardLayer_2_3/Tensordot/Const:output:0*
T0*
_output_shapes
: �
Lmodel_1/transformer_encoder_5/Encoder-FeedForwardLayer_2_3/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
Kmodel_1/transformer_encoder_5/Encoder-FeedForwardLayer_2_3/Tensordot/Prod_1ProdXmodel_1/transformer_encoder_5/Encoder-FeedForwardLayer_2_3/Tensordot/GatherV2_1:output:0Umodel_1/transformer_encoder_5/Encoder-FeedForwardLayer_2_3/Tensordot/Const_1:output:0*
T0*
_output_shapes
: �
Pmodel_1/transformer_encoder_5/Encoder-FeedForwardLayer_2_3/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Kmodel_1/transformer_encoder_5/Encoder-FeedForwardLayer_2_3/Tensordot/concatConcatV2Rmodel_1/transformer_encoder_5/Encoder-FeedForwardLayer_2_3/Tensordot/free:output:0Rmodel_1/transformer_encoder_5/Encoder-FeedForwardLayer_2_3/Tensordot/axes:output:0Ymodel_1/transformer_encoder_5/Encoder-FeedForwardLayer_2_3/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
Jmodel_1/transformer_encoder_5/Encoder-FeedForwardLayer_2_3/Tensordot/stackPackRmodel_1/transformer_encoder_5/Encoder-FeedForwardLayer_2_3/Tensordot/Prod:output:0Tmodel_1/transformer_encoder_5/Encoder-FeedForwardLayer_2_3/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
Nmodel_1/transformer_encoder_5/Encoder-FeedForwardLayer_2_3/Tensordot/transpose	TransposeImodel_1/transformer_encoder_5/Encoder-FeedForwardLayer_1_3/Gelu/mul_1:z:0Tmodel_1/transformer_encoder_5/Encoder-FeedForwardLayer_2_3/Tensordot/concat:output:0*
T0*+
_output_shapes
:���������P	�
Lmodel_1/transformer_encoder_5/Encoder-FeedForwardLayer_2_3/Tensordot/ReshapeReshapeRmodel_1/transformer_encoder_5/Encoder-FeedForwardLayer_2_3/Tensordot/transpose:y:0Smodel_1/transformer_encoder_5/Encoder-FeedForwardLayer_2_3/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
Kmodel_1/transformer_encoder_5/Encoder-FeedForwardLayer_2_3/Tensordot/MatMulMatMulUmodel_1/transformer_encoder_5/Encoder-FeedForwardLayer_2_3/Tensordot/Reshape:output:0[model_1/transformer_encoder_5/Encoder-FeedForwardLayer_2_3/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������	�
Lmodel_1/transformer_encoder_5/Encoder-FeedForwardLayer_2_3/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:	�
Rmodel_1/transformer_encoder_5/Encoder-FeedForwardLayer_2_3/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Mmodel_1/transformer_encoder_5/Encoder-FeedForwardLayer_2_3/Tensordot/concat_1ConcatV2Vmodel_1/transformer_encoder_5/Encoder-FeedForwardLayer_2_3/Tensordot/GatherV2:output:0Umodel_1/transformer_encoder_5/Encoder-FeedForwardLayer_2_3/Tensordot/Const_2:output:0[model_1/transformer_encoder_5/Encoder-FeedForwardLayer_2_3/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
Dmodel_1/transformer_encoder_5/Encoder-FeedForwardLayer_2_3/TensordotReshapeUmodel_1/transformer_encoder_5/Encoder-FeedForwardLayer_2_3/Tensordot/MatMul:product:0Vmodel_1/transformer_encoder_5/Encoder-FeedForwardLayer_2_3/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������P	�
Qmodel_1/transformer_encoder_5/Encoder-FeedForwardLayer_2_3/BiasAdd/ReadVariableOpReadVariableOpZmodel_1_transformer_encoder_5_encoder_feedforwardlayer_2_3_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype0�
Bmodel_1/transformer_encoder_5/Encoder-FeedForwardLayer_2_3/BiasAddBiasAddMmodel_1/transformer_encoder_5/Encoder-FeedForwardLayer_2_3/Tensordot:output:0Ymodel_1/transformer_encoder_5/Encoder-FeedForwardLayer_2_3/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������P	�
0model_1/transformer_encoder_5/dropout_5/IdentityIdentityKmodel_1/transformer_encoder_5/Encoder-FeedForwardLayer_2_3/BiasAdd:output:0*
T0*+
_output_shapes
:���������P	�
=model_1/transformer_encoder_5/Encoder-2nd-AdditionLayer-3/addAddV2Amodel_1/transformer_encoder_5/Encoder-1st-AdditionLayer-3/add:z:09model_1/transformer_encoder_5/dropout_5/Identity:output:0*
T0*+
_output_shapes
:���������P	�
model_1/FinalLayerNorm/ShapeShapeAmodel_1/transformer_encoder_5/Encoder-2nd-AdditionLayer-3/add:z:0*
T0*
_output_shapes
::��t
*model_1/FinalLayerNorm/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: v
,model_1/FinalLayerNorm/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:v
,model_1/FinalLayerNorm/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
$model_1/FinalLayerNorm/strided_sliceStridedSlice%model_1/FinalLayerNorm/Shape:output:03model_1/FinalLayerNorm/strided_slice/stack:output:05model_1/FinalLayerNorm/strided_slice/stack_1:output:05model_1/FinalLayerNorm/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskf
model_1/FinalLayerNorm/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
model_1/FinalLayerNorm/ProdProd-model_1/FinalLayerNorm/strided_slice:output:0%model_1/FinalLayerNorm/Const:output:0*
T0*
_output_shapes
: v
,model_1/FinalLayerNorm/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:x
.model_1/FinalLayerNorm/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: x
.model_1/FinalLayerNorm/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
&model_1/FinalLayerNorm/strided_slice_1StridedSlice%model_1/FinalLayerNorm/Shape:output:05model_1/FinalLayerNorm/strided_slice_1/stack:output:07model_1/FinalLayerNorm/strided_slice_1/stack_1:output:07model_1/FinalLayerNorm/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_maskh
model_1/FinalLayerNorm/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
model_1/FinalLayerNorm/Prod_1Prod/model_1/FinalLayerNorm/strided_slice_1:output:0'model_1/FinalLayerNorm/Const_1:output:0*
T0*
_output_shapes
: h
&model_1/FinalLayerNorm/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :h
&model_1/FinalLayerNorm/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :�
$model_1/FinalLayerNorm/Reshape/shapePack/model_1/FinalLayerNorm/Reshape/shape/0:output:0$model_1/FinalLayerNorm/Prod:output:0&model_1/FinalLayerNorm/Prod_1:output:0/model_1/FinalLayerNorm/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:�
model_1/FinalLayerNorm/ReshapeReshapeAmodel_1/transformer_encoder_5/Encoder-2nd-AdditionLayer-3/add:z:0-model_1/FinalLayerNorm/Reshape/shape:output:0*
T0*8
_output_shapes&
$:"������������������~
"model_1/FinalLayerNorm/ones/packedPack$model_1/FinalLayerNorm/Prod:output:0*
N*
T0*
_output_shapes
:f
!model_1/FinalLayerNorm/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
model_1/FinalLayerNorm/onesFill+model_1/FinalLayerNorm/ones/packed:output:0*model_1/FinalLayerNorm/ones/Const:output:0*
T0*#
_output_shapes
:���������
#model_1/FinalLayerNorm/zeros/packedPack$model_1/FinalLayerNorm/Prod:output:0*
N*
T0*
_output_shapes
:g
"model_1/FinalLayerNorm/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
model_1/FinalLayerNorm/zerosFill,model_1/FinalLayerNorm/zeros/packed:output:0+model_1/FinalLayerNorm/zeros/Const:output:0*
T0*#
_output_shapes
:���������a
model_1/FinalLayerNorm/Const_2Const*
_output_shapes
: *
dtype0*
valueB a
model_1/FinalLayerNorm/Const_3Const*
_output_shapes
: *
dtype0*
valueB �
'model_1/FinalLayerNorm/FusedBatchNormV3FusedBatchNormV3'model_1/FinalLayerNorm/Reshape:output:0$model_1/FinalLayerNorm/ones:output:0%model_1/FinalLayerNorm/zeros:output:0'model_1/FinalLayerNorm/Const_2:output:0'model_1/FinalLayerNorm/Const_3:output:0*
T0*
U0*x
_output_shapesf
d:"������������������:���������:���������:���������:���������:*
data_formatNCHW*
epsilon%o�:�
 model_1/FinalLayerNorm/Reshape_1Reshape+model_1/FinalLayerNorm/FusedBatchNormV3:y:0%model_1/FinalLayerNorm/Shape:output:0*
T0*+
_output_shapes
:���������P	�
)model_1/FinalLayerNorm/mul/ReadVariableOpReadVariableOp2model_1_finallayernorm_mul_readvariableop_resource*
_output_shapes
:	*
dtype0�
model_1/FinalLayerNorm/mulMul)model_1/FinalLayerNorm/Reshape_1:output:01model_1/FinalLayerNorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������P	�
)model_1/FinalLayerNorm/add/ReadVariableOpReadVariableOp2model_1_finallayernorm_add_readvariableop_resource*
_output_shapes
:	*
dtype0�
model_1/FinalLayerNorm/addAddV2model_1/FinalLayerNorm/mul:z:01model_1/FinalLayerNorm/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������P	�
>model_1/ReduceStackDimensionViaSummation/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :�
,model_1/ReduceStackDimensionViaSummation/SumSummodel_1/FinalLayerNorm/add:z:0Gmodel_1/ReduceStackDimensionViaSummation/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:���������	w
2model_1/ReduceStackDimensionViaSummation/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �B�
0model_1/ReduceStackDimensionViaSummation/truedivRealDiv5model_1/ReduceStackDimensionViaSummation/Sum:output:0;model_1/ReduceStackDimensionViaSummation/truediv/y:output:0*
T0*'
_output_shapes
:���������	z
!model_1/StandardizeTimeLimit/CastCasttimelimitinput*

DstT0*

SrcT0*'
_output_shapes
:���������g
"model_1/StandardizeTimeLimit/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  pA�
 model_1/StandardizeTimeLimit/subSub%model_1/StandardizeTimeLimit/Cast:y:0+model_1/StandardizeTimeLimit/sub/y:output:0*
T0*'
_output_shapes
:���������k
&model_1/StandardizeTimeLimit/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �@�
$model_1/StandardizeTimeLimit/truedivRealDiv$model_1/StandardizeTimeLimit/sub:z:0/model_1/StandardizeTimeLimit/truediv/y:output:0*
T0*'
_output_shapes
:���������f
$model_1/ConcatenateLayer/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
model_1/ConcatenateLayer/concatConcatV24model_1/ReduceStackDimensionViaSummation/truediv:z:0(model_1/StandardizeTimeLimit/truediv:z:0-model_1/ConcatenateLayer/concat/axis:output:0*
N*
T0*'
_output_shapes
:���������
�
<model_1/FullyConnectedLayerImprovement/MatMul/ReadVariableOpReadVariableOpEmodel_1_fullyconnectedlayerimprovement_matmul_readvariableop_resource*
_output_shapes

:

*
dtype0�
-model_1/FullyConnectedLayerImprovement/MatMulMatMul(model_1/ConcatenateLayer/concat:output:0Dmodel_1/FullyConnectedLayerImprovement/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
�
=model_1/FullyConnectedLayerImprovement/BiasAdd/ReadVariableOpReadVariableOpFmodel_1_fullyconnectedlayerimprovement_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0�
.model_1/FullyConnectedLayerImprovement/BiasAddBiasAdd7model_1/FullyConnectedLayerImprovement/MatMul:product:0Emodel_1/FullyConnectedLayerImprovement/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
v
1model_1/FullyConnectedLayerImprovement/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
/model_1/FullyConnectedLayerImprovement/Gelu/mulMul:model_1/FullyConnectedLayerImprovement/Gelu/mul/x:output:07model_1/FullyConnectedLayerImprovement/BiasAdd:output:0*
T0*'
_output_shapes
:���������
w
2model_1/FullyConnectedLayerImprovement/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *��?�
3model_1/FullyConnectedLayerImprovement/Gelu/truedivRealDiv7model_1/FullyConnectedLayerImprovement/BiasAdd:output:0;model_1/FullyConnectedLayerImprovement/Gelu/Cast/x:output:0*
T0*'
_output_shapes
:���������
�
/model_1/FullyConnectedLayerImprovement/Gelu/ErfErf7model_1/FullyConnectedLayerImprovement/Gelu/truediv:z:0*
T0*'
_output_shapes
:���������
v
1model_1/FullyConnectedLayerImprovement/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
/model_1/FullyConnectedLayerImprovement/Gelu/addAddV2:model_1/FullyConnectedLayerImprovement/Gelu/add/x:output:03model_1/FullyConnectedLayerImprovement/Gelu/Erf:y:0*
T0*'
_output_shapes
:���������
�
1model_1/FullyConnectedLayerImprovement/Gelu/mul_1Mul3model_1/FullyConnectedLayerImprovement/Gelu/mul:z:03model_1/FullyConnectedLayerImprovement/Gelu/add:z:0*
T0*'
_output_shapes
:���������
�
3model_1/PredictionImprovement/MatMul/ReadVariableOpReadVariableOp<model_1_predictionimprovement_matmul_readvariableop_resource*
_output_shapes

:
*
dtype0�
$model_1/PredictionImprovement/MatMulMatMul5model_1/FullyConnectedLayerImprovement/Gelu/mul_1:z:0;model_1/PredictionImprovement/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
4model_1/PredictionImprovement/BiasAdd/ReadVariableOpReadVariableOp=model_1_predictionimprovement_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
%model_1/PredictionImprovement/BiasAddBiasAdd.model_1/PredictionImprovement/MatMul:product:0<model_1/PredictionImprovement/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
%model_1/PredictionImprovement/SigmoidSigmoid.model_1/PredictionImprovement/BiasAdd:output:0*
T0*'
_output_shapes
:���������o
model_1/Output/SqueezeSqueeze)model_1/PredictionImprovement/Sigmoid:y:0*
T0*
_output_shapes
:_
IdentityIdentitymodel_1/Output/Squeeze:output:0^NoOp*
T0*
_output_shapes
:�$
NoOpNoOp*^model_1/FinalLayerNorm/add/ReadVariableOp*^model_1/FinalLayerNorm/mul/ReadVariableOp>^model_1/FullyConnectedLayerImprovement/BiasAdd/ReadVariableOp=^model_1/FullyConnectedLayerImprovement/MatMul/ReadVariableOp5^model_1/PredictionImprovement/BiasAdd/ReadVariableOp4^model_1/PredictionImprovement/MatMul/ReadVariableOpR^model_1/transformer_encoder_3/Encoder-1st-NormalizationLayer-1/add/ReadVariableOpR^model_1/transformer_encoder_3/Encoder-1st-NormalizationLayer-1/mul/ReadVariableOpR^model_1/transformer_encoder_3/Encoder-2nd-NormalizationLayer-1/add/ReadVariableOpR^model_1/transformer_encoder_3/Encoder-2nd-NormalizationLayer-1/mul/ReadVariableOpR^model_1/transformer_encoder_3/Encoder-FeedForwardLayer_1_1/BiasAdd/ReadVariableOpT^model_1/transformer_encoder_3/Encoder-FeedForwardLayer_1_1/Tensordot/ReadVariableOpR^model_1/transformer_encoder_3/Encoder-FeedForwardLayer_2_1/BiasAdd/ReadVariableOpT^model_1/transformer_encoder_3/Encoder-FeedForwardLayer_2_1/Tensordot/ReadVariableOp_^model_1/transformer_encoder_3/Encoder-SelfAttentionLayer-1/attention_output/add/ReadVariableOpi^model_1/transformer_encoder_3/Encoder-SelfAttentionLayer-1/attention_output/einsum/Einsum/ReadVariableOpR^model_1/transformer_encoder_3/Encoder-SelfAttentionLayer-1/key/add/ReadVariableOp\^model_1/transformer_encoder_3/Encoder-SelfAttentionLayer-1/key/einsum/Einsum/ReadVariableOpT^model_1/transformer_encoder_3/Encoder-SelfAttentionLayer-1/query/add/ReadVariableOp^^model_1/transformer_encoder_3/Encoder-SelfAttentionLayer-1/query/einsum/Einsum/ReadVariableOpT^model_1/transformer_encoder_3/Encoder-SelfAttentionLayer-1/value/add/ReadVariableOp^^model_1/transformer_encoder_3/Encoder-SelfAttentionLayer-1/value/einsum/Einsum/ReadVariableOpR^model_1/transformer_encoder_4/Encoder-1st-NormalizationLayer-2/add/ReadVariableOpR^model_1/transformer_encoder_4/Encoder-1st-NormalizationLayer-2/mul/ReadVariableOpR^model_1/transformer_encoder_4/Encoder-2nd-NormalizationLayer-2/add/ReadVariableOpR^model_1/transformer_encoder_4/Encoder-2nd-NormalizationLayer-2/mul/ReadVariableOpR^model_1/transformer_encoder_4/Encoder-FeedForwardLayer_1_2/BiasAdd/ReadVariableOpT^model_1/transformer_encoder_4/Encoder-FeedForwardLayer_1_2/Tensordot/ReadVariableOpR^model_1/transformer_encoder_4/Encoder-FeedForwardLayer_2_2/BiasAdd/ReadVariableOpT^model_1/transformer_encoder_4/Encoder-FeedForwardLayer_2_2/Tensordot/ReadVariableOp_^model_1/transformer_encoder_4/Encoder-SelfAttentionLayer-2/attention_output/add/ReadVariableOpi^model_1/transformer_encoder_4/Encoder-SelfAttentionLayer-2/attention_output/einsum/Einsum/ReadVariableOpR^model_1/transformer_encoder_4/Encoder-SelfAttentionLayer-2/key/add/ReadVariableOp\^model_1/transformer_encoder_4/Encoder-SelfAttentionLayer-2/key/einsum/Einsum/ReadVariableOpT^model_1/transformer_encoder_4/Encoder-SelfAttentionLayer-2/query/add/ReadVariableOp^^model_1/transformer_encoder_4/Encoder-SelfAttentionLayer-2/query/einsum/Einsum/ReadVariableOpT^model_1/transformer_encoder_4/Encoder-SelfAttentionLayer-2/value/add/ReadVariableOp^^model_1/transformer_encoder_4/Encoder-SelfAttentionLayer-2/value/einsum/Einsum/ReadVariableOpR^model_1/transformer_encoder_5/Encoder-1st-NormalizationLayer-3/add/ReadVariableOpR^model_1/transformer_encoder_5/Encoder-1st-NormalizationLayer-3/mul/ReadVariableOpR^model_1/transformer_encoder_5/Encoder-2nd-NormalizationLayer-3/add/ReadVariableOpR^model_1/transformer_encoder_5/Encoder-2nd-NormalizationLayer-3/mul/ReadVariableOpR^model_1/transformer_encoder_5/Encoder-FeedForwardLayer_1_3/BiasAdd/ReadVariableOpT^model_1/transformer_encoder_5/Encoder-FeedForwardLayer_1_3/Tensordot/ReadVariableOpR^model_1/transformer_encoder_5/Encoder-FeedForwardLayer_2_3/BiasAdd/ReadVariableOpT^model_1/transformer_encoder_5/Encoder-FeedForwardLayer_2_3/Tensordot/ReadVariableOp_^model_1/transformer_encoder_5/Encoder-SelfAttentionLayer-3/attention_output/add/ReadVariableOpi^model_1/transformer_encoder_5/Encoder-SelfAttentionLayer-3/attention_output/einsum/Einsum/ReadVariableOpR^model_1/transformer_encoder_5/Encoder-SelfAttentionLayer-3/key/add/ReadVariableOp\^model_1/transformer_encoder_5/Encoder-SelfAttentionLayer-3/key/einsum/Einsum/ReadVariableOpT^model_1/transformer_encoder_5/Encoder-SelfAttentionLayer-3/query/add/ReadVariableOp^^model_1/transformer_encoder_5/Encoder-SelfAttentionLayer-3/query/einsum/Einsum/ReadVariableOpT^model_1/transformer_encoder_5/Encoder-SelfAttentionLayer-3/value/add/ReadVariableOp^^model_1/transformer_encoder_5/Encoder-SelfAttentionLayer-3/value/einsum/Einsum/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:���������P	:���������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2V
)model_1/FinalLayerNorm/add/ReadVariableOp)model_1/FinalLayerNorm/add/ReadVariableOp2V
)model_1/FinalLayerNorm/mul/ReadVariableOp)model_1/FinalLayerNorm/mul/ReadVariableOp2~
=model_1/FullyConnectedLayerImprovement/BiasAdd/ReadVariableOp=model_1/FullyConnectedLayerImprovement/BiasAdd/ReadVariableOp2|
<model_1/FullyConnectedLayerImprovement/MatMul/ReadVariableOp<model_1/FullyConnectedLayerImprovement/MatMul/ReadVariableOp2l
4model_1/PredictionImprovement/BiasAdd/ReadVariableOp4model_1/PredictionImprovement/BiasAdd/ReadVariableOp2j
3model_1/PredictionImprovement/MatMul/ReadVariableOp3model_1/PredictionImprovement/MatMul/ReadVariableOp2�
Qmodel_1/transformer_encoder_3/Encoder-1st-NormalizationLayer-1/add/ReadVariableOpQmodel_1/transformer_encoder_3/Encoder-1st-NormalizationLayer-1/add/ReadVariableOp2�
Qmodel_1/transformer_encoder_3/Encoder-1st-NormalizationLayer-1/mul/ReadVariableOpQmodel_1/transformer_encoder_3/Encoder-1st-NormalizationLayer-1/mul/ReadVariableOp2�
Qmodel_1/transformer_encoder_3/Encoder-2nd-NormalizationLayer-1/add/ReadVariableOpQmodel_1/transformer_encoder_3/Encoder-2nd-NormalizationLayer-1/add/ReadVariableOp2�
Qmodel_1/transformer_encoder_3/Encoder-2nd-NormalizationLayer-1/mul/ReadVariableOpQmodel_1/transformer_encoder_3/Encoder-2nd-NormalizationLayer-1/mul/ReadVariableOp2�
Qmodel_1/transformer_encoder_3/Encoder-FeedForwardLayer_1_1/BiasAdd/ReadVariableOpQmodel_1/transformer_encoder_3/Encoder-FeedForwardLayer_1_1/BiasAdd/ReadVariableOp2�
Smodel_1/transformer_encoder_3/Encoder-FeedForwardLayer_1_1/Tensordot/ReadVariableOpSmodel_1/transformer_encoder_3/Encoder-FeedForwardLayer_1_1/Tensordot/ReadVariableOp2�
Qmodel_1/transformer_encoder_3/Encoder-FeedForwardLayer_2_1/BiasAdd/ReadVariableOpQmodel_1/transformer_encoder_3/Encoder-FeedForwardLayer_2_1/BiasAdd/ReadVariableOp2�
Smodel_1/transformer_encoder_3/Encoder-FeedForwardLayer_2_1/Tensordot/ReadVariableOpSmodel_1/transformer_encoder_3/Encoder-FeedForwardLayer_2_1/Tensordot/ReadVariableOp2�
^model_1/transformer_encoder_3/Encoder-SelfAttentionLayer-1/attention_output/add/ReadVariableOp^model_1/transformer_encoder_3/Encoder-SelfAttentionLayer-1/attention_output/add/ReadVariableOp2�
hmodel_1/transformer_encoder_3/Encoder-SelfAttentionLayer-1/attention_output/einsum/Einsum/ReadVariableOphmodel_1/transformer_encoder_3/Encoder-SelfAttentionLayer-1/attention_output/einsum/Einsum/ReadVariableOp2�
Qmodel_1/transformer_encoder_3/Encoder-SelfAttentionLayer-1/key/add/ReadVariableOpQmodel_1/transformer_encoder_3/Encoder-SelfAttentionLayer-1/key/add/ReadVariableOp2�
[model_1/transformer_encoder_3/Encoder-SelfAttentionLayer-1/key/einsum/Einsum/ReadVariableOp[model_1/transformer_encoder_3/Encoder-SelfAttentionLayer-1/key/einsum/Einsum/ReadVariableOp2�
Smodel_1/transformer_encoder_3/Encoder-SelfAttentionLayer-1/query/add/ReadVariableOpSmodel_1/transformer_encoder_3/Encoder-SelfAttentionLayer-1/query/add/ReadVariableOp2�
]model_1/transformer_encoder_3/Encoder-SelfAttentionLayer-1/query/einsum/Einsum/ReadVariableOp]model_1/transformer_encoder_3/Encoder-SelfAttentionLayer-1/query/einsum/Einsum/ReadVariableOp2�
Smodel_1/transformer_encoder_3/Encoder-SelfAttentionLayer-1/value/add/ReadVariableOpSmodel_1/transformer_encoder_3/Encoder-SelfAttentionLayer-1/value/add/ReadVariableOp2�
]model_1/transformer_encoder_3/Encoder-SelfAttentionLayer-1/value/einsum/Einsum/ReadVariableOp]model_1/transformer_encoder_3/Encoder-SelfAttentionLayer-1/value/einsum/Einsum/ReadVariableOp2�
Qmodel_1/transformer_encoder_4/Encoder-1st-NormalizationLayer-2/add/ReadVariableOpQmodel_1/transformer_encoder_4/Encoder-1st-NormalizationLayer-2/add/ReadVariableOp2�
Qmodel_1/transformer_encoder_4/Encoder-1st-NormalizationLayer-2/mul/ReadVariableOpQmodel_1/transformer_encoder_4/Encoder-1st-NormalizationLayer-2/mul/ReadVariableOp2�
Qmodel_1/transformer_encoder_4/Encoder-2nd-NormalizationLayer-2/add/ReadVariableOpQmodel_1/transformer_encoder_4/Encoder-2nd-NormalizationLayer-2/add/ReadVariableOp2�
Qmodel_1/transformer_encoder_4/Encoder-2nd-NormalizationLayer-2/mul/ReadVariableOpQmodel_1/transformer_encoder_4/Encoder-2nd-NormalizationLayer-2/mul/ReadVariableOp2�
Qmodel_1/transformer_encoder_4/Encoder-FeedForwardLayer_1_2/BiasAdd/ReadVariableOpQmodel_1/transformer_encoder_4/Encoder-FeedForwardLayer_1_2/BiasAdd/ReadVariableOp2�
Smodel_1/transformer_encoder_4/Encoder-FeedForwardLayer_1_2/Tensordot/ReadVariableOpSmodel_1/transformer_encoder_4/Encoder-FeedForwardLayer_1_2/Tensordot/ReadVariableOp2�
Qmodel_1/transformer_encoder_4/Encoder-FeedForwardLayer_2_2/BiasAdd/ReadVariableOpQmodel_1/transformer_encoder_4/Encoder-FeedForwardLayer_2_2/BiasAdd/ReadVariableOp2�
Smodel_1/transformer_encoder_4/Encoder-FeedForwardLayer_2_2/Tensordot/ReadVariableOpSmodel_1/transformer_encoder_4/Encoder-FeedForwardLayer_2_2/Tensordot/ReadVariableOp2�
^model_1/transformer_encoder_4/Encoder-SelfAttentionLayer-2/attention_output/add/ReadVariableOp^model_1/transformer_encoder_4/Encoder-SelfAttentionLayer-2/attention_output/add/ReadVariableOp2�
hmodel_1/transformer_encoder_4/Encoder-SelfAttentionLayer-2/attention_output/einsum/Einsum/ReadVariableOphmodel_1/transformer_encoder_4/Encoder-SelfAttentionLayer-2/attention_output/einsum/Einsum/ReadVariableOp2�
Qmodel_1/transformer_encoder_4/Encoder-SelfAttentionLayer-2/key/add/ReadVariableOpQmodel_1/transformer_encoder_4/Encoder-SelfAttentionLayer-2/key/add/ReadVariableOp2�
[model_1/transformer_encoder_4/Encoder-SelfAttentionLayer-2/key/einsum/Einsum/ReadVariableOp[model_1/transformer_encoder_4/Encoder-SelfAttentionLayer-2/key/einsum/Einsum/ReadVariableOp2�
Smodel_1/transformer_encoder_4/Encoder-SelfAttentionLayer-2/query/add/ReadVariableOpSmodel_1/transformer_encoder_4/Encoder-SelfAttentionLayer-2/query/add/ReadVariableOp2�
]model_1/transformer_encoder_4/Encoder-SelfAttentionLayer-2/query/einsum/Einsum/ReadVariableOp]model_1/transformer_encoder_4/Encoder-SelfAttentionLayer-2/query/einsum/Einsum/ReadVariableOp2�
Smodel_1/transformer_encoder_4/Encoder-SelfAttentionLayer-2/value/add/ReadVariableOpSmodel_1/transformer_encoder_4/Encoder-SelfAttentionLayer-2/value/add/ReadVariableOp2�
]model_1/transformer_encoder_4/Encoder-SelfAttentionLayer-2/value/einsum/Einsum/ReadVariableOp]model_1/transformer_encoder_4/Encoder-SelfAttentionLayer-2/value/einsum/Einsum/ReadVariableOp2�
Qmodel_1/transformer_encoder_5/Encoder-1st-NormalizationLayer-3/add/ReadVariableOpQmodel_1/transformer_encoder_5/Encoder-1st-NormalizationLayer-3/add/ReadVariableOp2�
Qmodel_1/transformer_encoder_5/Encoder-1st-NormalizationLayer-3/mul/ReadVariableOpQmodel_1/transformer_encoder_5/Encoder-1st-NormalizationLayer-3/mul/ReadVariableOp2�
Qmodel_1/transformer_encoder_5/Encoder-2nd-NormalizationLayer-3/add/ReadVariableOpQmodel_1/transformer_encoder_5/Encoder-2nd-NormalizationLayer-3/add/ReadVariableOp2�
Qmodel_1/transformer_encoder_5/Encoder-2nd-NormalizationLayer-3/mul/ReadVariableOpQmodel_1/transformer_encoder_5/Encoder-2nd-NormalizationLayer-3/mul/ReadVariableOp2�
Qmodel_1/transformer_encoder_5/Encoder-FeedForwardLayer_1_3/BiasAdd/ReadVariableOpQmodel_1/transformer_encoder_5/Encoder-FeedForwardLayer_1_3/BiasAdd/ReadVariableOp2�
Smodel_1/transformer_encoder_5/Encoder-FeedForwardLayer_1_3/Tensordot/ReadVariableOpSmodel_1/transformer_encoder_5/Encoder-FeedForwardLayer_1_3/Tensordot/ReadVariableOp2�
Qmodel_1/transformer_encoder_5/Encoder-FeedForwardLayer_2_3/BiasAdd/ReadVariableOpQmodel_1/transformer_encoder_5/Encoder-FeedForwardLayer_2_3/BiasAdd/ReadVariableOp2�
Smodel_1/transformer_encoder_5/Encoder-FeedForwardLayer_2_3/Tensordot/ReadVariableOpSmodel_1/transformer_encoder_5/Encoder-FeedForwardLayer_2_3/Tensordot/ReadVariableOp2�
^model_1/transformer_encoder_5/Encoder-SelfAttentionLayer-3/attention_output/add/ReadVariableOp^model_1/transformer_encoder_5/Encoder-SelfAttentionLayer-3/attention_output/add/ReadVariableOp2�
hmodel_1/transformer_encoder_5/Encoder-SelfAttentionLayer-3/attention_output/einsum/Einsum/ReadVariableOphmodel_1/transformer_encoder_5/Encoder-SelfAttentionLayer-3/attention_output/einsum/Einsum/ReadVariableOp2�
Qmodel_1/transformer_encoder_5/Encoder-SelfAttentionLayer-3/key/add/ReadVariableOpQmodel_1/transformer_encoder_5/Encoder-SelfAttentionLayer-3/key/add/ReadVariableOp2�
[model_1/transformer_encoder_5/Encoder-SelfAttentionLayer-3/key/einsum/Einsum/ReadVariableOp[model_1/transformer_encoder_5/Encoder-SelfAttentionLayer-3/key/einsum/Einsum/ReadVariableOp2�
Smodel_1/transformer_encoder_5/Encoder-SelfAttentionLayer-3/query/add/ReadVariableOpSmodel_1/transformer_encoder_5/Encoder-SelfAttentionLayer-3/query/add/ReadVariableOp2�
]model_1/transformer_encoder_5/Encoder-SelfAttentionLayer-3/query/einsum/Einsum/ReadVariableOp]model_1/transformer_encoder_5/Encoder-SelfAttentionLayer-3/query/einsum/Einsum/ReadVariableOp2�
Smodel_1/transformer_encoder_5/Encoder-SelfAttentionLayer-3/value/add/ReadVariableOpSmodel_1/transformer_encoder_5/Encoder-SelfAttentionLayer-3/value/add/ReadVariableOp2�
]model_1/transformer_encoder_5/Encoder-SelfAttentionLayer-3/value/einsum/Einsum/ReadVariableOp]model_1/transformer_encoder_5/Encoder-SelfAttentionLayer-3/value/einsum/Einsum/ReadVariableOp:(7$
"
_user_specified_name
resource:(6$
"
_user_specified_name
resource:(5$
"
_user_specified_name
resource:(4$
"
_user_specified_name
resource:(3$
"
_user_specified_name
resource:(2$
"
_user_specified_name
resource:(1$
"
_user_specified_name
resource:(0$
"
_user_specified_name
resource:(/$
"
_user_specified_name
resource:(.$
"
_user_specified_name
resource:(-$
"
_user_specified_name
resource:(,$
"
_user_specified_name
resource:(+$
"
_user_specified_name
resource:(*$
"
_user_specified_name
resource:()$
"
_user_specified_name
resource:(($
"
_user_specified_name
resource:('$
"
_user_specified_name
resource:(&$
"
_user_specified_name
resource:(%$
"
_user_specified_name
resource:($$
"
_user_specified_name
resource:(#$
"
_user_specified_name
resource:("$
"
_user_specified_name
resource:(!$
"
_user_specified_name
resource:( $
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:(
$
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:WS
'
_output_shapes
:���������
(
_user_specified_nameTimeLimitInput:d `
+
_output_shapes
:���������P	
1
_user_specified_nameStackLevelInputFeatures
�
l
P__inference_StandardizeTimeLimit_layer_call_and_return_conditional_losses_230500

inputs
identityJ
sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  pAT
subSubinputssub/y:output:0*
T0*'
_output_shapes
:���������N
	truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �@a
truedivRealDivsub:z:0truediv/y:output:0*
T0*'
_output_shapes
:���������S
IdentityIdentitytruediv:z:0*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
��
�
Q__inference_transformer_encoder_5_layer_call_and_return_conditional_losses_230401

inputsJ
<encoder_1st_normalizationlayer_3_mul_readvariableop_resource:	J
<encoder_1st_normalizationlayer_3_add_readvariableop_resource:	^
Hencoder_selfattentionlayer_3_query_einsum_einsum_readvariableop_resource:	P
>encoder_selfattentionlayer_3_query_add_readvariableop_resource:\
Fencoder_selfattentionlayer_3_key_einsum_einsum_readvariableop_resource:	N
<encoder_selfattentionlayer_3_key_add_readvariableop_resource:^
Hencoder_selfattentionlayer_3_value_einsum_einsum_readvariableop_resource:	P
>encoder_selfattentionlayer_3_value_add_readvariableop_resource:i
Sencoder_selfattentionlayer_3_attention_output_einsum_einsum_readvariableop_resource:	W
Iencoder_selfattentionlayer_3_attention_output_add_readvariableop_resource:	J
<encoder_2nd_normalizationlayer_3_mul_readvariableop_resource:	J
<encoder_2nd_normalizationlayer_3_add_readvariableop_resource:	P
>encoder_feedforwardlayer_1_3_tensordot_readvariableop_resource:		J
<encoder_feedforwardlayer_1_3_biasadd_readvariableop_resource:	P
>encoder_feedforwardlayer_2_3_tensordot_readvariableop_resource:		J
<encoder_feedforwardlayer_2_3_biasadd_readvariableop_resource:	
identity

identity_1��3Encoder-1st-NormalizationLayer-3/add/ReadVariableOp�3Encoder-1st-NormalizationLayer-3/mul/ReadVariableOp�3Encoder-2nd-NormalizationLayer-3/add/ReadVariableOp�3Encoder-2nd-NormalizationLayer-3/mul/ReadVariableOp�3Encoder-FeedForwardLayer_1_3/BiasAdd/ReadVariableOp�5Encoder-FeedForwardLayer_1_3/Tensordot/ReadVariableOp�3Encoder-FeedForwardLayer_2_3/BiasAdd/ReadVariableOp�5Encoder-FeedForwardLayer_2_3/Tensordot/ReadVariableOp�@Encoder-SelfAttentionLayer-3/attention_output/add/ReadVariableOp�JEncoder-SelfAttentionLayer-3/attention_output/einsum/Einsum/ReadVariableOp�3Encoder-SelfAttentionLayer-3/key/add/ReadVariableOp�=Encoder-SelfAttentionLayer-3/key/einsum/Einsum/ReadVariableOp�5Encoder-SelfAttentionLayer-3/query/add/ReadVariableOp�?Encoder-SelfAttentionLayer-3/query/einsum/Einsum/ReadVariableOp�5Encoder-SelfAttentionLayer-3/value/add/ReadVariableOp�?Encoder-SelfAttentionLayer-3/value/einsum/Einsum/ReadVariableOpj
&Encoder-1st-NormalizationLayer-3/ShapeShapeinputs*
T0*
_output_shapes
::��~
4Encoder-1st-NormalizationLayer-3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
6Encoder-1st-NormalizationLayer-3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
6Encoder-1st-NormalizationLayer-3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
.Encoder-1st-NormalizationLayer-3/strided_sliceStridedSlice/Encoder-1st-NormalizationLayer-3/Shape:output:0=Encoder-1st-NormalizationLayer-3/strided_slice/stack:output:0?Encoder-1st-NormalizationLayer-3/strided_slice/stack_1:output:0?Encoder-1st-NormalizationLayer-3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskp
&Encoder-1st-NormalizationLayer-3/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
%Encoder-1st-NormalizationLayer-3/ProdProd7Encoder-1st-NormalizationLayer-3/strided_slice:output:0/Encoder-1st-NormalizationLayer-3/Const:output:0*
T0*
_output_shapes
: �
6Encoder-1st-NormalizationLayer-3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:�
8Encoder-1st-NormalizationLayer-3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: �
8Encoder-1st-NormalizationLayer-3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
0Encoder-1st-NormalizationLayer-3/strided_slice_1StridedSlice/Encoder-1st-NormalizationLayer-3/Shape:output:0?Encoder-1st-NormalizationLayer-3/strided_slice_1/stack:output:0AEncoder-1st-NormalizationLayer-3/strided_slice_1/stack_1:output:0AEncoder-1st-NormalizationLayer-3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_maskr
(Encoder-1st-NormalizationLayer-3/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
'Encoder-1st-NormalizationLayer-3/Prod_1Prod9Encoder-1st-NormalizationLayer-3/strided_slice_1:output:01Encoder-1st-NormalizationLayer-3/Const_1:output:0*
T0*
_output_shapes
: r
0Encoder-1st-NormalizationLayer-3/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :r
0Encoder-1st-NormalizationLayer-3/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :�
.Encoder-1st-NormalizationLayer-3/Reshape/shapePack9Encoder-1st-NormalizationLayer-3/Reshape/shape/0:output:0.Encoder-1st-NormalizationLayer-3/Prod:output:00Encoder-1st-NormalizationLayer-3/Prod_1:output:09Encoder-1st-NormalizationLayer-3/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:�
(Encoder-1st-NormalizationLayer-3/ReshapeReshapeinputs7Encoder-1st-NormalizationLayer-3/Reshape/shape:output:0*
T0*8
_output_shapes&
$:"�������������������
,Encoder-1st-NormalizationLayer-3/ones/packedPack.Encoder-1st-NormalizationLayer-3/Prod:output:0*
N*
T0*
_output_shapes
:p
+Encoder-1st-NormalizationLayer-3/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
%Encoder-1st-NormalizationLayer-3/onesFill5Encoder-1st-NormalizationLayer-3/ones/packed:output:04Encoder-1st-NormalizationLayer-3/ones/Const:output:0*
T0*#
_output_shapes
:����������
-Encoder-1st-NormalizationLayer-3/zeros/packedPack.Encoder-1st-NormalizationLayer-3/Prod:output:0*
N*
T0*
_output_shapes
:q
,Encoder-1st-NormalizationLayer-3/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
&Encoder-1st-NormalizationLayer-3/zerosFill6Encoder-1st-NormalizationLayer-3/zeros/packed:output:05Encoder-1st-NormalizationLayer-3/zeros/Const:output:0*
T0*#
_output_shapes
:���������k
(Encoder-1st-NormalizationLayer-3/Const_2Const*
_output_shapes
: *
dtype0*
valueB k
(Encoder-1st-NormalizationLayer-3/Const_3Const*
_output_shapes
: *
dtype0*
valueB �
1Encoder-1st-NormalizationLayer-3/FusedBatchNormV3FusedBatchNormV31Encoder-1st-NormalizationLayer-3/Reshape:output:0.Encoder-1st-NormalizationLayer-3/ones:output:0/Encoder-1st-NormalizationLayer-3/zeros:output:01Encoder-1st-NormalizationLayer-3/Const_2:output:01Encoder-1st-NormalizationLayer-3/Const_3:output:0*
T0*
U0*x
_output_shapesf
d:"������������������:���������:���������:���������:���������:*
data_formatNCHW*
epsilon%o�:�
*Encoder-1st-NormalizationLayer-3/Reshape_1Reshape5Encoder-1st-NormalizationLayer-3/FusedBatchNormV3:y:0/Encoder-1st-NormalizationLayer-3/Shape:output:0*
T0*+
_output_shapes
:���������P	�
3Encoder-1st-NormalizationLayer-3/mul/ReadVariableOpReadVariableOp<encoder_1st_normalizationlayer_3_mul_readvariableop_resource*
_output_shapes
:	*
dtype0�
$Encoder-1st-NormalizationLayer-3/mulMul3Encoder-1st-NormalizationLayer-3/Reshape_1:output:0;Encoder-1st-NormalizationLayer-3/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������P	�
3Encoder-1st-NormalizationLayer-3/add/ReadVariableOpReadVariableOp<encoder_1st_normalizationlayer_3_add_readvariableop_resource*
_output_shapes
:	*
dtype0�
$Encoder-1st-NormalizationLayer-3/addAddV2(Encoder-1st-NormalizationLayer-3/mul:z:0;Encoder-1st-NormalizationLayer-3/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������P	�
?Encoder-SelfAttentionLayer-3/query/einsum/Einsum/ReadVariableOpReadVariableOpHencoder_selfattentionlayer_3_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:	*
dtype0�
0Encoder-SelfAttentionLayer-3/query/einsum/EinsumEinsum(Encoder-1st-NormalizationLayer-3/add:z:0GEncoder-SelfAttentionLayer-3/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������P*
equationabc,cde->abde�
5Encoder-SelfAttentionLayer-3/query/add/ReadVariableOpReadVariableOp>encoder_selfattentionlayer_3_query_add_readvariableop_resource*
_output_shapes

:*
dtype0�
&Encoder-SelfAttentionLayer-3/query/addAddV29Encoder-SelfAttentionLayer-3/query/einsum/Einsum:output:0=Encoder-SelfAttentionLayer-3/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������P�
=Encoder-SelfAttentionLayer-3/key/einsum/Einsum/ReadVariableOpReadVariableOpFencoder_selfattentionlayer_3_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:	*
dtype0�
.Encoder-SelfAttentionLayer-3/key/einsum/EinsumEinsum(Encoder-1st-NormalizationLayer-3/add:z:0EEncoder-SelfAttentionLayer-3/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������P*
equationabc,cde->abde�
3Encoder-SelfAttentionLayer-3/key/add/ReadVariableOpReadVariableOp<encoder_selfattentionlayer_3_key_add_readvariableop_resource*
_output_shapes

:*
dtype0�
$Encoder-SelfAttentionLayer-3/key/addAddV27Encoder-SelfAttentionLayer-3/key/einsum/Einsum:output:0;Encoder-SelfAttentionLayer-3/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������P�
?Encoder-SelfAttentionLayer-3/value/einsum/Einsum/ReadVariableOpReadVariableOpHencoder_selfattentionlayer_3_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:	*
dtype0�
0Encoder-SelfAttentionLayer-3/value/einsum/EinsumEinsum(Encoder-1st-NormalizationLayer-3/add:z:0GEncoder-SelfAttentionLayer-3/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������P*
equationabc,cde->abde�
5Encoder-SelfAttentionLayer-3/value/add/ReadVariableOpReadVariableOp>encoder_selfattentionlayer_3_value_add_readvariableop_resource*
_output_shapes

:*
dtype0�
&Encoder-SelfAttentionLayer-3/value/addAddV29Encoder-SelfAttentionLayer-3/value/einsum/Einsum:output:0=Encoder-SelfAttentionLayer-3/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������Pg
"Encoder-SelfAttentionLayer-3/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *:�?�
 Encoder-SelfAttentionLayer-3/MulMul*Encoder-SelfAttentionLayer-3/query/add:z:0+Encoder-SelfAttentionLayer-3/Mul/y:output:0*
T0*/
_output_shapes
:���������P�
*Encoder-SelfAttentionLayer-3/einsum/EinsumEinsum(Encoder-SelfAttentionLayer-3/key/add:z:0$Encoder-SelfAttentionLayer-3/Mul:z:0*
N*
T0*/
_output_shapes
:���������PP*
equationaecd,abcd->acbe�
,Encoder-SelfAttentionLayer-3/softmax/SoftmaxSoftmax3Encoder-SelfAttentionLayer-3/einsum/Einsum:output:0*
T0*/
_output_shapes
:���������PPw
2Encoder-SelfAttentionLayer-3/dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?�
0Encoder-SelfAttentionLayer-3/dropout/dropout/MulMul6Encoder-SelfAttentionLayer-3/softmax/Softmax:softmax:0;Encoder-SelfAttentionLayer-3/dropout/dropout/Const:output:0*
T0*/
_output_shapes
:���������PP�
2Encoder-SelfAttentionLayer-3/dropout/dropout/ShapeShape6Encoder-SelfAttentionLayer-3/softmax/Softmax:softmax:0*
T0*
_output_shapes
::���
IEncoder-SelfAttentionLayer-3/dropout/dropout/random_uniform/RandomUniformRandomUniform;Encoder-SelfAttentionLayer-3/dropout/dropout/Shape:output:0*
T0*/
_output_shapes
:���������PP*
dtype0*
seed���
;Encoder-SelfAttentionLayer-3/dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=�
9Encoder-SelfAttentionLayer-3/dropout/dropout/GreaterEqualGreaterEqualREncoder-SelfAttentionLayer-3/dropout/dropout/random_uniform/RandomUniform:output:0DEncoder-SelfAttentionLayer-3/dropout/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:���������PPy
4Encoder-SelfAttentionLayer-3/dropout/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
5Encoder-SelfAttentionLayer-3/dropout/dropout/SelectV2SelectV2=Encoder-SelfAttentionLayer-3/dropout/dropout/GreaterEqual:z:04Encoder-SelfAttentionLayer-3/dropout/dropout/Mul:z:0=Encoder-SelfAttentionLayer-3/dropout/dropout/Const_1:output:0*
T0*/
_output_shapes
:���������PP�
,Encoder-SelfAttentionLayer-3/einsum_1/EinsumEinsum>Encoder-SelfAttentionLayer-3/dropout/dropout/SelectV2:output:0*Encoder-SelfAttentionLayer-3/value/add:z:0*
N*
T0*/
_output_shapes
:���������P*
equationacbe,aecd->abcd�
JEncoder-SelfAttentionLayer-3/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpSencoder_selfattentionlayer_3_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:	*
dtype0�
;Encoder-SelfAttentionLayer-3/attention_output/einsum/EinsumEinsum5Encoder-SelfAttentionLayer-3/einsum_1/Einsum:output:0REncoder-SelfAttentionLayer-3/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:���������P	*
equationabcd,cde->abe�
@Encoder-SelfAttentionLayer-3/attention_output/add/ReadVariableOpReadVariableOpIencoder_selfattentionlayer_3_attention_output_add_readvariableop_resource*
_output_shapes
:	*
dtype0�
1Encoder-SelfAttentionLayer-3/attention_output/addAddV2DEncoder-SelfAttentionLayer-3/attention_output/einsum/Einsum:output:0HEncoder-SelfAttentionLayer-3/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������P	�
Encoder-1st-AdditionLayer-3/addAddV2inputs5Encoder-SelfAttentionLayer-3/attention_output/add:z:0*
T0*+
_output_shapes
:���������P	�
&Encoder-2nd-NormalizationLayer-3/ShapeShape#Encoder-1st-AdditionLayer-3/add:z:0*
T0*
_output_shapes
::��~
4Encoder-2nd-NormalizationLayer-3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
6Encoder-2nd-NormalizationLayer-3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
6Encoder-2nd-NormalizationLayer-3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
.Encoder-2nd-NormalizationLayer-3/strided_sliceStridedSlice/Encoder-2nd-NormalizationLayer-3/Shape:output:0=Encoder-2nd-NormalizationLayer-3/strided_slice/stack:output:0?Encoder-2nd-NormalizationLayer-3/strided_slice/stack_1:output:0?Encoder-2nd-NormalizationLayer-3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskp
&Encoder-2nd-NormalizationLayer-3/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
%Encoder-2nd-NormalizationLayer-3/ProdProd7Encoder-2nd-NormalizationLayer-3/strided_slice:output:0/Encoder-2nd-NormalizationLayer-3/Const:output:0*
T0*
_output_shapes
: �
6Encoder-2nd-NormalizationLayer-3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:�
8Encoder-2nd-NormalizationLayer-3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: �
8Encoder-2nd-NormalizationLayer-3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
0Encoder-2nd-NormalizationLayer-3/strided_slice_1StridedSlice/Encoder-2nd-NormalizationLayer-3/Shape:output:0?Encoder-2nd-NormalizationLayer-3/strided_slice_1/stack:output:0AEncoder-2nd-NormalizationLayer-3/strided_slice_1/stack_1:output:0AEncoder-2nd-NormalizationLayer-3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_maskr
(Encoder-2nd-NormalizationLayer-3/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
'Encoder-2nd-NormalizationLayer-3/Prod_1Prod9Encoder-2nd-NormalizationLayer-3/strided_slice_1:output:01Encoder-2nd-NormalizationLayer-3/Const_1:output:0*
T0*
_output_shapes
: r
0Encoder-2nd-NormalizationLayer-3/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :r
0Encoder-2nd-NormalizationLayer-3/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :�
.Encoder-2nd-NormalizationLayer-3/Reshape/shapePack9Encoder-2nd-NormalizationLayer-3/Reshape/shape/0:output:0.Encoder-2nd-NormalizationLayer-3/Prod:output:00Encoder-2nd-NormalizationLayer-3/Prod_1:output:09Encoder-2nd-NormalizationLayer-3/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:�
(Encoder-2nd-NormalizationLayer-3/ReshapeReshape#Encoder-1st-AdditionLayer-3/add:z:07Encoder-2nd-NormalizationLayer-3/Reshape/shape:output:0*
T0*8
_output_shapes&
$:"�������������������
,Encoder-2nd-NormalizationLayer-3/ones/packedPack.Encoder-2nd-NormalizationLayer-3/Prod:output:0*
N*
T0*
_output_shapes
:p
+Encoder-2nd-NormalizationLayer-3/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
%Encoder-2nd-NormalizationLayer-3/onesFill5Encoder-2nd-NormalizationLayer-3/ones/packed:output:04Encoder-2nd-NormalizationLayer-3/ones/Const:output:0*
T0*#
_output_shapes
:����������
-Encoder-2nd-NormalizationLayer-3/zeros/packedPack.Encoder-2nd-NormalizationLayer-3/Prod:output:0*
N*
T0*
_output_shapes
:q
,Encoder-2nd-NormalizationLayer-3/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
&Encoder-2nd-NormalizationLayer-3/zerosFill6Encoder-2nd-NormalizationLayer-3/zeros/packed:output:05Encoder-2nd-NormalizationLayer-3/zeros/Const:output:0*
T0*#
_output_shapes
:���������k
(Encoder-2nd-NormalizationLayer-3/Const_2Const*
_output_shapes
: *
dtype0*
valueB k
(Encoder-2nd-NormalizationLayer-3/Const_3Const*
_output_shapes
: *
dtype0*
valueB �
1Encoder-2nd-NormalizationLayer-3/FusedBatchNormV3FusedBatchNormV31Encoder-2nd-NormalizationLayer-3/Reshape:output:0.Encoder-2nd-NormalizationLayer-3/ones:output:0/Encoder-2nd-NormalizationLayer-3/zeros:output:01Encoder-2nd-NormalizationLayer-3/Const_2:output:01Encoder-2nd-NormalizationLayer-3/Const_3:output:0*
T0*
U0*x
_output_shapesf
d:"������������������:���������:���������:���������:���������:*
data_formatNCHW*
epsilon%o�:�
*Encoder-2nd-NormalizationLayer-3/Reshape_1Reshape5Encoder-2nd-NormalizationLayer-3/FusedBatchNormV3:y:0/Encoder-2nd-NormalizationLayer-3/Shape:output:0*
T0*+
_output_shapes
:���������P	�
3Encoder-2nd-NormalizationLayer-3/mul/ReadVariableOpReadVariableOp<encoder_2nd_normalizationlayer_3_mul_readvariableop_resource*
_output_shapes
:	*
dtype0�
$Encoder-2nd-NormalizationLayer-3/mulMul3Encoder-2nd-NormalizationLayer-3/Reshape_1:output:0;Encoder-2nd-NormalizationLayer-3/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������P	�
3Encoder-2nd-NormalizationLayer-3/add/ReadVariableOpReadVariableOp<encoder_2nd_normalizationlayer_3_add_readvariableop_resource*
_output_shapes
:	*
dtype0�
$Encoder-2nd-NormalizationLayer-3/addAddV2(Encoder-2nd-NormalizationLayer-3/mul:z:0;Encoder-2nd-NormalizationLayer-3/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������P	�
5Encoder-FeedForwardLayer_1_3/Tensordot/ReadVariableOpReadVariableOp>encoder_feedforwardlayer_1_3_tensordot_readvariableop_resource*
_output_shapes

:		*
dtype0u
+Encoder-FeedForwardLayer_1_3/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:|
+Encoder-FeedForwardLayer_1_3/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       �
,Encoder-FeedForwardLayer_1_3/Tensordot/ShapeShape(Encoder-2nd-NormalizationLayer-3/add:z:0*
T0*
_output_shapes
::��v
4Encoder-FeedForwardLayer_1_3/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
/Encoder-FeedForwardLayer_1_3/Tensordot/GatherV2GatherV25Encoder-FeedForwardLayer_1_3/Tensordot/Shape:output:04Encoder-FeedForwardLayer_1_3/Tensordot/free:output:0=Encoder-FeedForwardLayer_1_3/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:x
6Encoder-FeedForwardLayer_1_3/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
1Encoder-FeedForwardLayer_1_3/Tensordot/GatherV2_1GatherV25Encoder-FeedForwardLayer_1_3/Tensordot/Shape:output:04Encoder-FeedForwardLayer_1_3/Tensordot/axes:output:0?Encoder-FeedForwardLayer_1_3/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:v
,Encoder-FeedForwardLayer_1_3/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
+Encoder-FeedForwardLayer_1_3/Tensordot/ProdProd8Encoder-FeedForwardLayer_1_3/Tensordot/GatherV2:output:05Encoder-FeedForwardLayer_1_3/Tensordot/Const:output:0*
T0*
_output_shapes
: x
.Encoder-FeedForwardLayer_1_3/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
-Encoder-FeedForwardLayer_1_3/Tensordot/Prod_1Prod:Encoder-FeedForwardLayer_1_3/Tensordot/GatherV2_1:output:07Encoder-FeedForwardLayer_1_3/Tensordot/Const_1:output:0*
T0*
_output_shapes
: t
2Encoder-FeedForwardLayer_1_3/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
-Encoder-FeedForwardLayer_1_3/Tensordot/concatConcatV24Encoder-FeedForwardLayer_1_3/Tensordot/free:output:04Encoder-FeedForwardLayer_1_3/Tensordot/axes:output:0;Encoder-FeedForwardLayer_1_3/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
,Encoder-FeedForwardLayer_1_3/Tensordot/stackPack4Encoder-FeedForwardLayer_1_3/Tensordot/Prod:output:06Encoder-FeedForwardLayer_1_3/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
0Encoder-FeedForwardLayer_1_3/Tensordot/transpose	Transpose(Encoder-2nd-NormalizationLayer-3/add:z:06Encoder-FeedForwardLayer_1_3/Tensordot/concat:output:0*
T0*+
_output_shapes
:���������P	�
.Encoder-FeedForwardLayer_1_3/Tensordot/ReshapeReshape4Encoder-FeedForwardLayer_1_3/Tensordot/transpose:y:05Encoder-FeedForwardLayer_1_3/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
-Encoder-FeedForwardLayer_1_3/Tensordot/MatMulMatMul7Encoder-FeedForwardLayer_1_3/Tensordot/Reshape:output:0=Encoder-FeedForwardLayer_1_3/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������	x
.Encoder-FeedForwardLayer_1_3/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:	v
4Encoder-FeedForwardLayer_1_3/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
/Encoder-FeedForwardLayer_1_3/Tensordot/concat_1ConcatV28Encoder-FeedForwardLayer_1_3/Tensordot/GatherV2:output:07Encoder-FeedForwardLayer_1_3/Tensordot/Const_2:output:0=Encoder-FeedForwardLayer_1_3/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
&Encoder-FeedForwardLayer_1_3/TensordotReshape7Encoder-FeedForwardLayer_1_3/Tensordot/MatMul:product:08Encoder-FeedForwardLayer_1_3/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������P	�
3Encoder-FeedForwardLayer_1_3/BiasAdd/ReadVariableOpReadVariableOp<encoder_feedforwardlayer_1_3_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype0�
$Encoder-FeedForwardLayer_1_3/BiasAddBiasAdd/Encoder-FeedForwardLayer_1_3/Tensordot:output:0;Encoder-FeedForwardLayer_1_3/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������P	l
'Encoder-FeedForwardLayer_1_3/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
%Encoder-FeedForwardLayer_1_3/Gelu/mulMul0Encoder-FeedForwardLayer_1_3/Gelu/mul/x:output:0-Encoder-FeedForwardLayer_1_3/BiasAdd:output:0*
T0*+
_output_shapes
:���������P	m
(Encoder-FeedForwardLayer_1_3/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *��?�
)Encoder-FeedForwardLayer_1_3/Gelu/truedivRealDiv-Encoder-FeedForwardLayer_1_3/BiasAdd:output:01Encoder-FeedForwardLayer_1_3/Gelu/Cast/x:output:0*
T0*+
_output_shapes
:���������P	�
%Encoder-FeedForwardLayer_1_3/Gelu/ErfErf-Encoder-FeedForwardLayer_1_3/Gelu/truediv:z:0*
T0*+
_output_shapes
:���������P	l
'Encoder-FeedForwardLayer_1_3/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
%Encoder-FeedForwardLayer_1_3/Gelu/addAddV20Encoder-FeedForwardLayer_1_3/Gelu/add/x:output:0)Encoder-FeedForwardLayer_1_3/Gelu/Erf:y:0*
T0*+
_output_shapes
:���������P	�
'Encoder-FeedForwardLayer_1_3/Gelu/mul_1Mul)Encoder-FeedForwardLayer_1_3/Gelu/mul:z:0)Encoder-FeedForwardLayer_1_3/Gelu/add:z:0*
T0*+
_output_shapes
:���������P	�
5Encoder-FeedForwardLayer_2_3/Tensordot/ReadVariableOpReadVariableOp>encoder_feedforwardlayer_2_3_tensordot_readvariableop_resource*
_output_shapes

:		*
dtype0u
+Encoder-FeedForwardLayer_2_3/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:|
+Encoder-FeedForwardLayer_2_3/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       �
,Encoder-FeedForwardLayer_2_3/Tensordot/ShapeShape+Encoder-FeedForwardLayer_1_3/Gelu/mul_1:z:0*
T0*
_output_shapes
::��v
4Encoder-FeedForwardLayer_2_3/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
/Encoder-FeedForwardLayer_2_3/Tensordot/GatherV2GatherV25Encoder-FeedForwardLayer_2_3/Tensordot/Shape:output:04Encoder-FeedForwardLayer_2_3/Tensordot/free:output:0=Encoder-FeedForwardLayer_2_3/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:x
6Encoder-FeedForwardLayer_2_3/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
1Encoder-FeedForwardLayer_2_3/Tensordot/GatherV2_1GatherV25Encoder-FeedForwardLayer_2_3/Tensordot/Shape:output:04Encoder-FeedForwardLayer_2_3/Tensordot/axes:output:0?Encoder-FeedForwardLayer_2_3/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:v
,Encoder-FeedForwardLayer_2_3/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
+Encoder-FeedForwardLayer_2_3/Tensordot/ProdProd8Encoder-FeedForwardLayer_2_3/Tensordot/GatherV2:output:05Encoder-FeedForwardLayer_2_3/Tensordot/Const:output:0*
T0*
_output_shapes
: x
.Encoder-FeedForwardLayer_2_3/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
-Encoder-FeedForwardLayer_2_3/Tensordot/Prod_1Prod:Encoder-FeedForwardLayer_2_3/Tensordot/GatherV2_1:output:07Encoder-FeedForwardLayer_2_3/Tensordot/Const_1:output:0*
T0*
_output_shapes
: t
2Encoder-FeedForwardLayer_2_3/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
-Encoder-FeedForwardLayer_2_3/Tensordot/concatConcatV24Encoder-FeedForwardLayer_2_3/Tensordot/free:output:04Encoder-FeedForwardLayer_2_3/Tensordot/axes:output:0;Encoder-FeedForwardLayer_2_3/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
,Encoder-FeedForwardLayer_2_3/Tensordot/stackPack4Encoder-FeedForwardLayer_2_3/Tensordot/Prod:output:06Encoder-FeedForwardLayer_2_3/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
0Encoder-FeedForwardLayer_2_3/Tensordot/transpose	Transpose+Encoder-FeedForwardLayer_1_3/Gelu/mul_1:z:06Encoder-FeedForwardLayer_2_3/Tensordot/concat:output:0*
T0*+
_output_shapes
:���������P	�
.Encoder-FeedForwardLayer_2_3/Tensordot/ReshapeReshape4Encoder-FeedForwardLayer_2_3/Tensordot/transpose:y:05Encoder-FeedForwardLayer_2_3/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
-Encoder-FeedForwardLayer_2_3/Tensordot/MatMulMatMul7Encoder-FeedForwardLayer_2_3/Tensordot/Reshape:output:0=Encoder-FeedForwardLayer_2_3/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������	x
.Encoder-FeedForwardLayer_2_3/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:	v
4Encoder-FeedForwardLayer_2_3/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
/Encoder-FeedForwardLayer_2_3/Tensordot/concat_1ConcatV28Encoder-FeedForwardLayer_2_3/Tensordot/GatherV2:output:07Encoder-FeedForwardLayer_2_3/Tensordot/Const_2:output:0=Encoder-FeedForwardLayer_2_3/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
&Encoder-FeedForwardLayer_2_3/TensordotReshape7Encoder-FeedForwardLayer_2_3/Tensordot/MatMul:product:08Encoder-FeedForwardLayer_2_3/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������P	�
3Encoder-FeedForwardLayer_2_3/BiasAdd/ReadVariableOpReadVariableOp<encoder_feedforwardlayer_2_3_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype0�
$Encoder-FeedForwardLayer_2_3/BiasAddBiasAdd/Encoder-FeedForwardLayer_2_3/Tensordot:output:0;Encoder-FeedForwardLayer_2_3/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������P	\
dropout_5/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?�
dropout_5/dropout/MulMul-Encoder-FeedForwardLayer_2_3/BiasAdd:output:0 dropout_5/dropout/Const:output:0*
T0*+
_output_shapes
:���������P	�
dropout_5/dropout/ShapeShape-Encoder-FeedForwardLayer_2_3/BiasAdd:output:0*
T0*
_output_shapes
::���
.dropout_5/dropout/random_uniform/RandomUniformRandomUniform dropout_5/dropout/Shape:output:0*
T0*+
_output_shapes
:���������P	*
dtype0*
seed2*
seed��e
 dropout_5/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=�
dropout_5/dropout/GreaterEqualGreaterEqual7dropout_5/dropout/random_uniform/RandomUniform:output:0)dropout_5/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:���������P	^
dropout_5/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout_5/dropout/SelectV2SelectV2"dropout_5/dropout/GreaterEqual:z:0dropout_5/dropout/Mul:z:0"dropout_5/dropout/Const_1:output:0*
T0*+
_output_shapes
:���������P	�
Encoder-2nd-AdditionLayer-3/addAddV2#Encoder-1st-AdditionLayer-3/add:z:0#dropout_5/dropout/SelectV2:output:0*
T0*+
_output_shapes
:���������P	v
IdentityIdentity#Encoder-2nd-AdditionLayer-3/add:z:0^NoOp*
T0*+
_output_shapes
:���������P	�

Identity_1Identity6Encoder-SelfAttentionLayer-3/softmax/Softmax:softmax:0^NoOp*
T0*/
_output_shapes
:���������PP�
NoOpNoOp4^Encoder-1st-NormalizationLayer-3/add/ReadVariableOp4^Encoder-1st-NormalizationLayer-3/mul/ReadVariableOp4^Encoder-2nd-NormalizationLayer-3/add/ReadVariableOp4^Encoder-2nd-NormalizationLayer-3/mul/ReadVariableOp4^Encoder-FeedForwardLayer_1_3/BiasAdd/ReadVariableOp6^Encoder-FeedForwardLayer_1_3/Tensordot/ReadVariableOp4^Encoder-FeedForwardLayer_2_3/BiasAdd/ReadVariableOp6^Encoder-FeedForwardLayer_2_3/Tensordot/ReadVariableOpA^Encoder-SelfAttentionLayer-3/attention_output/add/ReadVariableOpK^Encoder-SelfAttentionLayer-3/attention_output/einsum/Einsum/ReadVariableOp4^Encoder-SelfAttentionLayer-3/key/add/ReadVariableOp>^Encoder-SelfAttentionLayer-3/key/einsum/Einsum/ReadVariableOp6^Encoder-SelfAttentionLayer-3/query/add/ReadVariableOp@^Encoder-SelfAttentionLayer-3/query/einsum/Einsum/ReadVariableOp6^Encoder-SelfAttentionLayer-3/value/add/ReadVariableOp@^Encoder-SelfAttentionLayer-3/value/einsum/Einsum/ReadVariableOp*
_output_shapes
 "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:���������P	: : : : : : : : : : : : : : : : 2j
3Encoder-1st-NormalizationLayer-3/add/ReadVariableOp3Encoder-1st-NormalizationLayer-3/add/ReadVariableOp2j
3Encoder-1st-NormalizationLayer-3/mul/ReadVariableOp3Encoder-1st-NormalizationLayer-3/mul/ReadVariableOp2j
3Encoder-2nd-NormalizationLayer-3/add/ReadVariableOp3Encoder-2nd-NormalizationLayer-3/add/ReadVariableOp2j
3Encoder-2nd-NormalizationLayer-3/mul/ReadVariableOp3Encoder-2nd-NormalizationLayer-3/mul/ReadVariableOp2j
3Encoder-FeedForwardLayer_1_3/BiasAdd/ReadVariableOp3Encoder-FeedForwardLayer_1_3/BiasAdd/ReadVariableOp2n
5Encoder-FeedForwardLayer_1_3/Tensordot/ReadVariableOp5Encoder-FeedForwardLayer_1_3/Tensordot/ReadVariableOp2j
3Encoder-FeedForwardLayer_2_3/BiasAdd/ReadVariableOp3Encoder-FeedForwardLayer_2_3/BiasAdd/ReadVariableOp2n
5Encoder-FeedForwardLayer_2_3/Tensordot/ReadVariableOp5Encoder-FeedForwardLayer_2_3/Tensordot/ReadVariableOp2�
@Encoder-SelfAttentionLayer-3/attention_output/add/ReadVariableOp@Encoder-SelfAttentionLayer-3/attention_output/add/ReadVariableOp2�
JEncoder-SelfAttentionLayer-3/attention_output/einsum/Einsum/ReadVariableOpJEncoder-SelfAttentionLayer-3/attention_output/einsum/Einsum/ReadVariableOp2j
3Encoder-SelfAttentionLayer-3/key/add/ReadVariableOp3Encoder-SelfAttentionLayer-3/key/add/ReadVariableOp2~
=Encoder-SelfAttentionLayer-3/key/einsum/Einsum/ReadVariableOp=Encoder-SelfAttentionLayer-3/key/einsum/Einsum/ReadVariableOp2n
5Encoder-SelfAttentionLayer-3/query/add/ReadVariableOp5Encoder-SelfAttentionLayer-3/query/add/ReadVariableOp2�
?Encoder-SelfAttentionLayer-3/query/einsum/Einsum/ReadVariableOp?Encoder-SelfAttentionLayer-3/query/einsum/Einsum/ReadVariableOp2n
5Encoder-SelfAttentionLayer-3/value/add/ReadVariableOp5Encoder-SelfAttentionLayer-3/value/add/ReadVariableOp2�
?Encoder-SelfAttentionLayer-3/value/einsum/Einsum/ReadVariableOp?Encoder-SelfAttentionLayer-3/value/einsum/Einsum/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:(
$
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:S O
+
_output_shapes
:���������P	
 
_user_specified_nameinputs
�
x
\__inference_ReduceStackDimensionViaSummation_layer_call_and_return_conditional_losses_233286

inputs
identityW
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :d
SumSuminputsSum/reduction_indices:output:0*
T0*'
_output_shapes
:���������	N
	truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �Bf
truedivRealDivSum:output:0truediv/y:output:0*
T0*'
_output_shapes
:���������	S
IdentityIdentitytruediv:z:0*
T0*'
_output_shapes
:���������	"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������P	:S O
+
_output_shapes
:���������P	
 
_user_specified_nameinputs
��
�
Q__inference_transformer_encoder_3_layer_call_and_return_conditional_losses_232155

inputsJ
<encoder_1st_normalizationlayer_1_mul_readvariableop_resource:	J
<encoder_1st_normalizationlayer_1_add_readvariableop_resource:	^
Hencoder_selfattentionlayer_1_query_einsum_einsum_readvariableop_resource:	P
>encoder_selfattentionlayer_1_query_add_readvariableop_resource:\
Fencoder_selfattentionlayer_1_key_einsum_einsum_readvariableop_resource:	N
<encoder_selfattentionlayer_1_key_add_readvariableop_resource:^
Hencoder_selfattentionlayer_1_value_einsum_einsum_readvariableop_resource:	P
>encoder_selfattentionlayer_1_value_add_readvariableop_resource:i
Sencoder_selfattentionlayer_1_attention_output_einsum_einsum_readvariableop_resource:	W
Iencoder_selfattentionlayer_1_attention_output_add_readvariableop_resource:	J
<encoder_2nd_normalizationlayer_1_mul_readvariableop_resource:	J
<encoder_2nd_normalizationlayer_1_add_readvariableop_resource:	P
>encoder_feedforwardlayer_1_1_tensordot_readvariableop_resource:		J
<encoder_feedforwardlayer_1_1_biasadd_readvariableop_resource:	P
>encoder_feedforwardlayer_2_1_tensordot_readvariableop_resource:		J
<encoder_feedforwardlayer_2_1_biasadd_readvariableop_resource:	
identity

identity_1��3Encoder-1st-NormalizationLayer-1/add/ReadVariableOp�3Encoder-1st-NormalizationLayer-1/mul/ReadVariableOp�3Encoder-2nd-NormalizationLayer-1/add/ReadVariableOp�3Encoder-2nd-NormalizationLayer-1/mul/ReadVariableOp�3Encoder-FeedForwardLayer_1_1/BiasAdd/ReadVariableOp�5Encoder-FeedForwardLayer_1_1/Tensordot/ReadVariableOp�3Encoder-FeedForwardLayer_2_1/BiasAdd/ReadVariableOp�5Encoder-FeedForwardLayer_2_1/Tensordot/ReadVariableOp�@Encoder-SelfAttentionLayer-1/attention_output/add/ReadVariableOp�JEncoder-SelfAttentionLayer-1/attention_output/einsum/Einsum/ReadVariableOp�3Encoder-SelfAttentionLayer-1/key/add/ReadVariableOp�=Encoder-SelfAttentionLayer-1/key/einsum/Einsum/ReadVariableOp�5Encoder-SelfAttentionLayer-1/query/add/ReadVariableOp�?Encoder-SelfAttentionLayer-1/query/einsum/Einsum/ReadVariableOp�5Encoder-SelfAttentionLayer-1/value/add/ReadVariableOp�?Encoder-SelfAttentionLayer-1/value/einsum/Einsum/ReadVariableOpj
&Encoder-1st-NormalizationLayer-1/ShapeShapeinputs*
T0*
_output_shapes
::��~
4Encoder-1st-NormalizationLayer-1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
6Encoder-1st-NormalizationLayer-1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
6Encoder-1st-NormalizationLayer-1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
.Encoder-1st-NormalizationLayer-1/strided_sliceStridedSlice/Encoder-1st-NormalizationLayer-1/Shape:output:0=Encoder-1st-NormalizationLayer-1/strided_slice/stack:output:0?Encoder-1st-NormalizationLayer-1/strided_slice/stack_1:output:0?Encoder-1st-NormalizationLayer-1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskp
&Encoder-1st-NormalizationLayer-1/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
%Encoder-1st-NormalizationLayer-1/ProdProd7Encoder-1st-NormalizationLayer-1/strided_slice:output:0/Encoder-1st-NormalizationLayer-1/Const:output:0*
T0*
_output_shapes
: �
6Encoder-1st-NormalizationLayer-1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:�
8Encoder-1st-NormalizationLayer-1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: �
8Encoder-1st-NormalizationLayer-1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
0Encoder-1st-NormalizationLayer-1/strided_slice_1StridedSlice/Encoder-1st-NormalizationLayer-1/Shape:output:0?Encoder-1st-NormalizationLayer-1/strided_slice_1/stack:output:0AEncoder-1st-NormalizationLayer-1/strided_slice_1/stack_1:output:0AEncoder-1st-NormalizationLayer-1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_maskr
(Encoder-1st-NormalizationLayer-1/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
'Encoder-1st-NormalizationLayer-1/Prod_1Prod9Encoder-1st-NormalizationLayer-1/strided_slice_1:output:01Encoder-1st-NormalizationLayer-1/Const_1:output:0*
T0*
_output_shapes
: r
0Encoder-1st-NormalizationLayer-1/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :r
0Encoder-1st-NormalizationLayer-1/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :�
.Encoder-1st-NormalizationLayer-1/Reshape/shapePack9Encoder-1st-NormalizationLayer-1/Reshape/shape/0:output:0.Encoder-1st-NormalizationLayer-1/Prod:output:00Encoder-1st-NormalizationLayer-1/Prod_1:output:09Encoder-1st-NormalizationLayer-1/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:�
(Encoder-1st-NormalizationLayer-1/ReshapeReshapeinputs7Encoder-1st-NormalizationLayer-1/Reshape/shape:output:0*
T0*8
_output_shapes&
$:"�������������������
,Encoder-1st-NormalizationLayer-1/ones/packedPack.Encoder-1st-NormalizationLayer-1/Prod:output:0*
N*
T0*
_output_shapes
:p
+Encoder-1st-NormalizationLayer-1/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
%Encoder-1st-NormalizationLayer-1/onesFill5Encoder-1st-NormalizationLayer-1/ones/packed:output:04Encoder-1st-NormalizationLayer-1/ones/Const:output:0*
T0*#
_output_shapes
:����������
-Encoder-1st-NormalizationLayer-1/zeros/packedPack.Encoder-1st-NormalizationLayer-1/Prod:output:0*
N*
T0*
_output_shapes
:q
,Encoder-1st-NormalizationLayer-1/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
&Encoder-1st-NormalizationLayer-1/zerosFill6Encoder-1st-NormalizationLayer-1/zeros/packed:output:05Encoder-1st-NormalizationLayer-1/zeros/Const:output:0*
T0*#
_output_shapes
:���������k
(Encoder-1st-NormalizationLayer-1/Const_2Const*
_output_shapes
: *
dtype0*
valueB k
(Encoder-1st-NormalizationLayer-1/Const_3Const*
_output_shapes
: *
dtype0*
valueB �
1Encoder-1st-NormalizationLayer-1/FusedBatchNormV3FusedBatchNormV31Encoder-1st-NormalizationLayer-1/Reshape:output:0.Encoder-1st-NormalizationLayer-1/ones:output:0/Encoder-1st-NormalizationLayer-1/zeros:output:01Encoder-1st-NormalizationLayer-1/Const_2:output:01Encoder-1st-NormalizationLayer-1/Const_3:output:0*
T0*
U0*x
_output_shapesf
d:"������������������:���������:���������:���������:���������:*
data_formatNCHW*
epsilon%o�:�
*Encoder-1st-NormalizationLayer-1/Reshape_1Reshape5Encoder-1st-NormalizationLayer-1/FusedBatchNormV3:y:0/Encoder-1st-NormalizationLayer-1/Shape:output:0*
T0*+
_output_shapes
:���������P	�
3Encoder-1st-NormalizationLayer-1/mul/ReadVariableOpReadVariableOp<encoder_1st_normalizationlayer_1_mul_readvariableop_resource*
_output_shapes
:	*
dtype0�
$Encoder-1st-NormalizationLayer-1/mulMul3Encoder-1st-NormalizationLayer-1/Reshape_1:output:0;Encoder-1st-NormalizationLayer-1/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������P	�
3Encoder-1st-NormalizationLayer-1/add/ReadVariableOpReadVariableOp<encoder_1st_normalizationlayer_1_add_readvariableop_resource*
_output_shapes
:	*
dtype0�
$Encoder-1st-NormalizationLayer-1/addAddV2(Encoder-1st-NormalizationLayer-1/mul:z:0;Encoder-1st-NormalizationLayer-1/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������P	�
?Encoder-SelfAttentionLayer-1/query/einsum/Einsum/ReadVariableOpReadVariableOpHencoder_selfattentionlayer_1_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:	*
dtype0�
0Encoder-SelfAttentionLayer-1/query/einsum/EinsumEinsum(Encoder-1st-NormalizationLayer-1/add:z:0GEncoder-SelfAttentionLayer-1/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������P*
equationabc,cde->abde�
5Encoder-SelfAttentionLayer-1/query/add/ReadVariableOpReadVariableOp>encoder_selfattentionlayer_1_query_add_readvariableop_resource*
_output_shapes

:*
dtype0�
&Encoder-SelfAttentionLayer-1/query/addAddV29Encoder-SelfAttentionLayer-1/query/einsum/Einsum:output:0=Encoder-SelfAttentionLayer-1/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������P�
=Encoder-SelfAttentionLayer-1/key/einsum/Einsum/ReadVariableOpReadVariableOpFencoder_selfattentionlayer_1_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:	*
dtype0�
.Encoder-SelfAttentionLayer-1/key/einsum/EinsumEinsum(Encoder-1st-NormalizationLayer-1/add:z:0EEncoder-SelfAttentionLayer-1/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������P*
equationabc,cde->abde�
3Encoder-SelfAttentionLayer-1/key/add/ReadVariableOpReadVariableOp<encoder_selfattentionlayer_1_key_add_readvariableop_resource*
_output_shapes

:*
dtype0�
$Encoder-SelfAttentionLayer-1/key/addAddV27Encoder-SelfAttentionLayer-1/key/einsum/Einsum:output:0;Encoder-SelfAttentionLayer-1/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������P�
?Encoder-SelfAttentionLayer-1/value/einsum/Einsum/ReadVariableOpReadVariableOpHencoder_selfattentionlayer_1_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:	*
dtype0�
0Encoder-SelfAttentionLayer-1/value/einsum/EinsumEinsum(Encoder-1st-NormalizationLayer-1/add:z:0GEncoder-SelfAttentionLayer-1/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������P*
equationabc,cde->abde�
5Encoder-SelfAttentionLayer-1/value/add/ReadVariableOpReadVariableOp>encoder_selfattentionlayer_1_value_add_readvariableop_resource*
_output_shapes

:*
dtype0�
&Encoder-SelfAttentionLayer-1/value/addAddV29Encoder-SelfAttentionLayer-1/value/einsum/Einsum:output:0=Encoder-SelfAttentionLayer-1/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������Pg
"Encoder-SelfAttentionLayer-1/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *:�?�
 Encoder-SelfAttentionLayer-1/MulMul*Encoder-SelfAttentionLayer-1/query/add:z:0+Encoder-SelfAttentionLayer-1/Mul/y:output:0*
T0*/
_output_shapes
:���������P�
*Encoder-SelfAttentionLayer-1/einsum/EinsumEinsum(Encoder-SelfAttentionLayer-1/key/add:z:0$Encoder-SelfAttentionLayer-1/Mul:z:0*
N*
T0*/
_output_shapes
:���������PP*
equationaecd,abcd->acbe�
,Encoder-SelfAttentionLayer-1/softmax/SoftmaxSoftmax3Encoder-SelfAttentionLayer-1/einsum/Einsum:output:0*
T0*/
_output_shapes
:���������PPw
2Encoder-SelfAttentionLayer-1/dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?�
0Encoder-SelfAttentionLayer-1/dropout/dropout/MulMul6Encoder-SelfAttentionLayer-1/softmax/Softmax:softmax:0;Encoder-SelfAttentionLayer-1/dropout/dropout/Const:output:0*
T0*/
_output_shapes
:���������PP�
2Encoder-SelfAttentionLayer-1/dropout/dropout/ShapeShape6Encoder-SelfAttentionLayer-1/softmax/Softmax:softmax:0*
T0*
_output_shapes
::���
IEncoder-SelfAttentionLayer-1/dropout/dropout/random_uniform/RandomUniformRandomUniform;Encoder-SelfAttentionLayer-1/dropout/dropout/Shape:output:0*
T0*/
_output_shapes
:���������PP*
dtype0*
seed���
;Encoder-SelfAttentionLayer-1/dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=�
9Encoder-SelfAttentionLayer-1/dropout/dropout/GreaterEqualGreaterEqualREncoder-SelfAttentionLayer-1/dropout/dropout/random_uniform/RandomUniform:output:0DEncoder-SelfAttentionLayer-1/dropout/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:���������PPy
4Encoder-SelfAttentionLayer-1/dropout/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
5Encoder-SelfAttentionLayer-1/dropout/dropout/SelectV2SelectV2=Encoder-SelfAttentionLayer-1/dropout/dropout/GreaterEqual:z:04Encoder-SelfAttentionLayer-1/dropout/dropout/Mul:z:0=Encoder-SelfAttentionLayer-1/dropout/dropout/Const_1:output:0*
T0*/
_output_shapes
:���������PP�
,Encoder-SelfAttentionLayer-1/einsum_1/EinsumEinsum>Encoder-SelfAttentionLayer-1/dropout/dropout/SelectV2:output:0*Encoder-SelfAttentionLayer-1/value/add:z:0*
N*
T0*/
_output_shapes
:���������P*
equationacbe,aecd->abcd�
JEncoder-SelfAttentionLayer-1/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpSencoder_selfattentionlayer_1_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:	*
dtype0�
;Encoder-SelfAttentionLayer-1/attention_output/einsum/EinsumEinsum5Encoder-SelfAttentionLayer-1/einsum_1/Einsum:output:0REncoder-SelfAttentionLayer-1/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:���������P	*
equationabcd,cde->abe�
@Encoder-SelfAttentionLayer-1/attention_output/add/ReadVariableOpReadVariableOpIencoder_selfattentionlayer_1_attention_output_add_readvariableop_resource*
_output_shapes
:	*
dtype0�
1Encoder-SelfAttentionLayer-1/attention_output/addAddV2DEncoder-SelfAttentionLayer-1/attention_output/einsum/Einsum:output:0HEncoder-SelfAttentionLayer-1/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������P	�
Encoder-1st-AdditionLayer-1/addAddV2inputs5Encoder-SelfAttentionLayer-1/attention_output/add:z:0*
T0*+
_output_shapes
:���������P	�
&Encoder-2nd-NormalizationLayer-1/ShapeShape#Encoder-1st-AdditionLayer-1/add:z:0*
T0*
_output_shapes
::��~
4Encoder-2nd-NormalizationLayer-1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
6Encoder-2nd-NormalizationLayer-1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
6Encoder-2nd-NormalizationLayer-1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
.Encoder-2nd-NormalizationLayer-1/strided_sliceStridedSlice/Encoder-2nd-NormalizationLayer-1/Shape:output:0=Encoder-2nd-NormalizationLayer-1/strided_slice/stack:output:0?Encoder-2nd-NormalizationLayer-1/strided_slice/stack_1:output:0?Encoder-2nd-NormalizationLayer-1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskp
&Encoder-2nd-NormalizationLayer-1/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
%Encoder-2nd-NormalizationLayer-1/ProdProd7Encoder-2nd-NormalizationLayer-1/strided_slice:output:0/Encoder-2nd-NormalizationLayer-1/Const:output:0*
T0*
_output_shapes
: �
6Encoder-2nd-NormalizationLayer-1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:�
8Encoder-2nd-NormalizationLayer-1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: �
8Encoder-2nd-NormalizationLayer-1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
0Encoder-2nd-NormalizationLayer-1/strided_slice_1StridedSlice/Encoder-2nd-NormalizationLayer-1/Shape:output:0?Encoder-2nd-NormalizationLayer-1/strided_slice_1/stack:output:0AEncoder-2nd-NormalizationLayer-1/strided_slice_1/stack_1:output:0AEncoder-2nd-NormalizationLayer-1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_maskr
(Encoder-2nd-NormalizationLayer-1/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
'Encoder-2nd-NormalizationLayer-1/Prod_1Prod9Encoder-2nd-NormalizationLayer-1/strided_slice_1:output:01Encoder-2nd-NormalizationLayer-1/Const_1:output:0*
T0*
_output_shapes
: r
0Encoder-2nd-NormalizationLayer-1/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :r
0Encoder-2nd-NormalizationLayer-1/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :�
.Encoder-2nd-NormalizationLayer-1/Reshape/shapePack9Encoder-2nd-NormalizationLayer-1/Reshape/shape/0:output:0.Encoder-2nd-NormalizationLayer-1/Prod:output:00Encoder-2nd-NormalizationLayer-1/Prod_1:output:09Encoder-2nd-NormalizationLayer-1/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:�
(Encoder-2nd-NormalizationLayer-1/ReshapeReshape#Encoder-1st-AdditionLayer-1/add:z:07Encoder-2nd-NormalizationLayer-1/Reshape/shape:output:0*
T0*8
_output_shapes&
$:"�������������������
,Encoder-2nd-NormalizationLayer-1/ones/packedPack.Encoder-2nd-NormalizationLayer-1/Prod:output:0*
N*
T0*
_output_shapes
:p
+Encoder-2nd-NormalizationLayer-1/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
%Encoder-2nd-NormalizationLayer-1/onesFill5Encoder-2nd-NormalizationLayer-1/ones/packed:output:04Encoder-2nd-NormalizationLayer-1/ones/Const:output:0*
T0*#
_output_shapes
:����������
-Encoder-2nd-NormalizationLayer-1/zeros/packedPack.Encoder-2nd-NormalizationLayer-1/Prod:output:0*
N*
T0*
_output_shapes
:q
,Encoder-2nd-NormalizationLayer-1/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
&Encoder-2nd-NormalizationLayer-1/zerosFill6Encoder-2nd-NormalizationLayer-1/zeros/packed:output:05Encoder-2nd-NormalizationLayer-1/zeros/Const:output:0*
T0*#
_output_shapes
:���������k
(Encoder-2nd-NormalizationLayer-1/Const_2Const*
_output_shapes
: *
dtype0*
valueB k
(Encoder-2nd-NormalizationLayer-1/Const_3Const*
_output_shapes
: *
dtype0*
valueB �
1Encoder-2nd-NormalizationLayer-1/FusedBatchNormV3FusedBatchNormV31Encoder-2nd-NormalizationLayer-1/Reshape:output:0.Encoder-2nd-NormalizationLayer-1/ones:output:0/Encoder-2nd-NormalizationLayer-1/zeros:output:01Encoder-2nd-NormalizationLayer-1/Const_2:output:01Encoder-2nd-NormalizationLayer-1/Const_3:output:0*
T0*
U0*x
_output_shapesf
d:"������������������:���������:���������:���������:���������:*
data_formatNCHW*
epsilon%o�:�
*Encoder-2nd-NormalizationLayer-1/Reshape_1Reshape5Encoder-2nd-NormalizationLayer-1/FusedBatchNormV3:y:0/Encoder-2nd-NormalizationLayer-1/Shape:output:0*
T0*+
_output_shapes
:���������P	�
3Encoder-2nd-NormalizationLayer-1/mul/ReadVariableOpReadVariableOp<encoder_2nd_normalizationlayer_1_mul_readvariableop_resource*
_output_shapes
:	*
dtype0�
$Encoder-2nd-NormalizationLayer-1/mulMul3Encoder-2nd-NormalizationLayer-1/Reshape_1:output:0;Encoder-2nd-NormalizationLayer-1/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������P	�
3Encoder-2nd-NormalizationLayer-1/add/ReadVariableOpReadVariableOp<encoder_2nd_normalizationlayer_1_add_readvariableop_resource*
_output_shapes
:	*
dtype0�
$Encoder-2nd-NormalizationLayer-1/addAddV2(Encoder-2nd-NormalizationLayer-1/mul:z:0;Encoder-2nd-NormalizationLayer-1/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������P	�
5Encoder-FeedForwardLayer_1_1/Tensordot/ReadVariableOpReadVariableOp>encoder_feedforwardlayer_1_1_tensordot_readvariableop_resource*
_output_shapes

:		*
dtype0u
+Encoder-FeedForwardLayer_1_1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:|
+Encoder-FeedForwardLayer_1_1/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       �
,Encoder-FeedForwardLayer_1_1/Tensordot/ShapeShape(Encoder-2nd-NormalizationLayer-1/add:z:0*
T0*
_output_shapes
::��v
4Encoder-FeedForwardLayer_1_1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
/Encoder-FeedForwardLayer_1_1/Tensordot/GatherV2GatherV25Encoder-FeedForwardLayer_1_1/Tensordot/Shape:output:04Encoder-FeedForwardLayer_1_1/Tensordot/free:output:0=Encoder-FeedForwardLayer_1_1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:x
6Encoder-FeedForwardLayer_1_1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
1Encoder-FeedForwardLayer_1_1/Tensordot/GatherV2_1GatherV25Encoder-FeedForwardLayer_1_1/Tensordot/Shape:output:04Encoder-FeedForwardLayer_1_1/Tensordot/axes:output:0?Encoder-FeedForwardLayer_1_1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:v
,Encoder-FeedForwardLayer_1_1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
+Encoder-FeedForwardLayer_1_1/Tensordot/ProdProd8Encoder-FeedForwardLayer_1_1/Tensordot/GatherV2:output:05Encoder-FeedForwardLayer_1_1/Tensordot/Const:output:0*
T0*
_output_shapes
: x
.Encoder-FeedForwardLayer_1_1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
-Encoder-FeedForwardLayer_1_1/Tensordot/Prod_1Prod:Encoder-FeedForwardLayer_1_1/Tensordot/GatherV2_1:output:07Encoder-FeedForwardLayer_1_1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: t
2Encoder-FeedForwardLayer_1_1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
-Encoder-FeedForwardLayer_1_1/Tensordot/concatConcatV24Encoder-FeedForwardLayer_1_1/Tensordot/free:output:04Encoder-FeedForwardLayer_1_1/Tensordot/axes:output:0;Encoder-FeedForwardLayer_1_1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
,Encoder-FeedForwardLayer_1_1/Tensordot/stackPack4Encoder-FeedForwardLayer_1_1/Tensordot/Prod:output:06Encoder-FeedForwardLayer_1_1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
0Encoder-FeedForwardLayer_1_1/Tensordot/transpose	Transpose(Encoder-2nd-NormalizationLayer-1/add:z:06Encoder-FeedForwardLayer_1_1/Tensordot/concat:output:0*
T0*+
_output_shapes
:���������P	�
.Encoder-FeedForwardLayer_1_1/Tensordot/ReshapeReshape4Encoder-FeedForwardLayer_1_1/Tensordot/transpose:y:05Encoder-FeedForwardLayer_1_1/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
-Encoder-FeedForwardLayer_1_1/Tensordot/MatMulMatMul7Encoder-FeedForwardLayer_1_1/Tensordot/Reshape:output:0=Encoder-FeedForwardLayer_1_1/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������	x
.Encoder-FeedForwardLayer_1_1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:	v
4Encoder-FeedForwardLayer_1_1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
/Encoder-FeedForwardLayer_1_1/Tensordot/concat_1ConcatV28Encoder-FeedForwardLayer_1_1/Tensordot/GatherV2:output:07Encoder-FeedForwardLayer_1_1/Tensordot/Const_2:output:0=Encoder-FeedForwardLayer_1_1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
&Encoder-FeedForwardLayer_1_1/TensordotReshape7Encoder-FeedForwardLayer_1_1/Tensordot/MatMul:product:08Encoder-FeedForwardLayer_1_1/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������P	�
3Encoder-FeedForwardLayer_1_1/BiasAdd/ReadVariableOpReadVariableOp<encoder_feedforwardlayer_1_1_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype0�
$Encoder-FeedForwardLayer_1_1/BiasAddBiasAdd/Encoder-FeedForwardLayer_1_1/Tensordot:output:0;Encoder-FeedForwardLayer_1_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������P	l
'Encoder-FeedForwardLayer_1_1/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
%Encoder-FeedForwardLayer_1_1/Gelu/mulMul0Encoder-FeedForwardLayer_1_1/Gelu/mul/x:output:0-Encoder-FeedForwardLayer_1_1/BiasAdd:output:0*
T0*+
_output_shapes
:���������P	m
(Encoder-FeedForwardLayer_1_1/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *��?�
)Encoder-FeedForwardLayer_1_1/Gelu/truedivRealDiv-Encoder-FeedForwardLayer_1_1/BiasAdd:output:01Encoder-FeedForwardLayer_1_1/Gelu/Cast/x:output:0*
T0*+
_output_shapes
:���������P	�
%Encoder-FeedForwardLayer_1_1/Gelu/ErfErf-Encoder-FeedForwardLayer_1_1/Gelu/truediv:z:0*
T0*+
_output_shapes
:���������P	l
'Encoder-FeedForwardLayer_1_1/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
%Encoder-FeedForwardLayer_1_1/Gelu/addAddV20Encoder-FeedForwardLayer_1_1/Gelu/add/x:output:0)Encoder-FeedForwardLayer_1_1/Gelu/Erf:y:0*
T0*+
_output_shapes
:���������P	�
'Encoder-FeedForwardLayer_1_1/Gelu/mul_1Mul)Encoder-FeedForwardLayer_1_1/Gelu/mul:z:0)Encoder-FeedForwardLayer_1_1/Gelu/add:z:0*
T0*+
_output_shapes
:���������P	�
5Encoder-FeedForwardLayer_2_1/Tensordot/ReadVariableOpReadVariableOp>encoder_feedforwardlayer_2_1_tensordot_readvariableop_resource*
_output_shapes

:		*
dtype0u
+Encoder-FeedForwardLayer_2_1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:|
+Encoder-FeedForwardLayer_2_1/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       �
,Encoder-FeedForwardLayer_2_1/Tensordot/ShapeShape+Encoder-FeedForwardLayer_1_1/Gelu/mul_1:z:0*
T0*
_output_shapes
::��v
4Encoder-FeedForwardLayer_2_1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
/Encoder-FeedForwardLayer_2_1/Tensordot/GatherV2GatherV25Encoder-FeedForwardLayer_2_1/Tensordot/Shape:output:04Encoder-FeedForwardLayer_2_1/Tensordot/free:output:0=Encoder-FeedForwardLayer_2_1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:x
6Encoder-FeedForwardLayer_2_1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
1Encoder-FeedForwardLayer_2_1/Tensordot/GatherV2_1GatherV25Encoder-FeedForwardLayer_2_1/Tensordot/Shape:output:04Encoder-FeedForwardLayer_2_1/Tensordot/axes:output:0?Encoder-FeedForwardLayer_2_1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:v
,Encoder-FeedForwardLayer_2_1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
+Encoder-FeedForwardLayer_2_1/Tensordot/ProdProd8Encoder-FeedForwardLayer_2_1/Tensordot/GatherV2:output:05Encoder-FeedForwardLayer_2_1/Tensordot/Const:output:0*
T0*
_output_shapes
: x
.Encoder-FeedForwardLayer_2_1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
-Encoder-FeedForwardLayer_2_1/Tensordot/Prod_1Prod:Encoder-FeedForwardLayer_2_1/Tensordot/GatherV2_1:output:07Encoder-FeedForwardLayer_2_1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: t
2Encoder-FeedForwardLayer_2_1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
-Encoder-FeedForwardLayer_2_1/Tensordot/concatConcatV24Encoder-FeedForwardLayer_2_1/Tensordot/free:output:04Encoder-FeedForwardLayer_2_1/Tensordot/axes:output:0;Encoder-FeedForwardLayer_2_1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
,Encoder-FeedForwardLayer_2_1/Tensordot/stackPack4Encoder-FeedForwardLayer_2_1/Tensordot/Prod:output:06Encoder-FeedForwardLayer_2_1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
0Encoder-FeedForwardLayer_2_1/Tensordot/transpose	Transpose+Encoder-FeedForwardLayer_1_1/Gelu/mul_1:z:06Encoder-FeedForwardLayer_2_1/Tensordot/concat:output:0*
T0*+
_output_shapes
:���������P	�
.Encoder-FeedForwardLayer_2_1/Tensordot/ReshapeReshape4Encoder-FeedForwardLayer_2_1/Tensordot/transpose:y:05Encoder-FeedForwardLayer_2_1/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
-Encoder-FeedForwardLayer_2_1/Tensordot/MatMulMatMul7Encoder-FeedForwardLayer_2_1/Tensordot/Reshape:output:0=Encoder-FeedForwardLayer_2_1/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������	x
.Encoder-FeedForwardLayer_2_1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:	v
4Encoder-FeedForwardLayer_2_1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
/Encoder-FeedForwardLayer_2_1/Tensordot/concat_1ConcatV28Encoder-FeedForwardLayer_2_1/Tensordot/GatherV2:output:07Encoder-FeedForwardLayer_2_1/Tensordot/Const_2:output:0=Encoder-FeedForwardLayer_2_1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
&Encoder-FeedForwardLayer_2_1/TensordotReshape7Encoder-FeedForwardLayer_2_1/Tensordot/MatMul:product:08Encoder-FeedForwardLayer_2_1/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������P	�
3Encoder-FeedForwardLayer_2_1/BiasAdd/ReadVariableOpReadVariableOp<encoder_feedforwardlayer_2_1_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype0�
$Encoder-FeedForwardLayer_2_1/BiasAddBiasAdd/Encoder-FeedForwardLayer_2_1/Tensordot:output:0;Encoder-FeedForwardLayer_2_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������P	\
dropout_3/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?�
dropout_3/dropout/MulMul-Encoder-FeedForwardLayer_2_1/BiasAdd:output:0 dropout_3/dropout/Const:output:0*
T0*+
_output_shapes
:���������P	�
dropout_3/dropout/ShapeShape-Encoder-FeedForwardLayer_2_1/BiasAdd:output:0*
T0*
_output_shapes
::���
.dropout_3/dropout/random_uniform/RandomUniformRandomUniform dropout_3/dropout/Shape:output:0*
T0*+
_output_shapes
:���������P	*
dtype0*
seed2*
seed��e
 dropout_3/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=�
dropout_3/dropout/GreaterEqualGreaterEqual7dropout_3/dropout/random_uniform/RandomUniform:output:0)dropout_3/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:���������P	^
dropout_3/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout_3/dropout/SelectV2SelectV2"dropout_3/dropout/GreaterEqual:z:0dropout_3/dropout/Mul:z:0"dropout_3/dropout/Const_1:output:0*
T0*+
_output_shapes
:���������P	�
Encoder-2nd-AdditionLayer-1/addAddV2#Encoder-1st-AdditionLayer-1/add:z:0#dropout_3/dropout/SelectV2:output:0*
T0*+
_output_shapes
:���������P	v
IdentityIdentity#Encoder-2nd-AdditionLayer-1/add:z:0^NoOp*
T0*+
_output_shapes
:���������P	�

Identity_1Identity6Encoder-SelfAttentionLayer-1/softmax/Softmax:softmax:0^NoOp*
T0*/
_output_shapes
:���������PP�
NoOpNoOp4^Encoder-1st-NormalizationLayer-1/add/ReadVariableOp4^Encoder-1st-NormalizationLayer-1/mul/ReadVariableOp4^Encoder-2nd-NormalizationLayer-1/add/ReadVariableOp4^Encoder-2nd-NormalizationLayer-1/mul/ReadVariableOp4^Encoder-FeedForwardLayer_1_1/BiasAdd/ReadVariableOp6^Encoder-FeedForwardLayer_1_1/Tensordot/ReadVariableOp4^Encoder-FeedForwardLayer_2_1/BiasAdd/ReadVariableOp6^Encoder-FeedForwardLayer_2_1/Tensordot/ReadVariableOpA^Encoder-SelfAttentionLayer-1/attention_output/add/ReadVariableOpK^Encoder-SelfAttentionLayer-1/attention_output/einsum/Einsum/ReadVariableOp4^Encoder-SelfAttentionLayer-1/key/add/ReadVariableOp>^Encoder-SelfAttentionLayer-1/key/einsum/Einsum/ReadVariableOp6^Encoder-SelfAttentionLayer-1/query/add/ReadVariableOp@^Encoder-SelfAttentionLayer-1/query/einsum/Einsum/ReadVariableOp6^Encoder-SelfAttentionLayer-1/value/add/ReadVariableOp@^Encoder-SelfAttentionLayer-1/value/einsum/Einsum/ReadVariableOp*
_output_shapes
 "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:���������P	: : : : : : : : : : : : : : : : 2j
3Encoder-1st-NormalizationLayer-1/add/ReadVariableOp3Encoder-1st-NormalizationLayer-1/add/ReadVariableOp2j
3Encoder-1st-NormalizationLayer-1/mul/ReadVariableOp3Encoder-1st-NormalizationLayer-1/mul/ReadVariableOp2j
3Encoder-2nd-NormalizationLayer-1/add/ReadVariableOp3Encoder-2nd-NormalizationLayer-1/add/ReadVariableOp2j
3Encoder-2nd-NormalizationLayer-1/mul/ReadVariableOp3Encoder-2nd-NormalizationLayer-1/mul/ReadVariableOp2j
3Encoder-FeedForwardLayer_1_1/BiasAdd/ReadVariableOp3Encoder-FeedForwardLayer_1_1/BiasAdd/ReadVariableOp2n
5Encoder-FeedForwardLayer_1_1/Tensordot/ReadVariableOp5Encoder-FeedForwardLayer_1_1/Tensordot/ReadVariableOp2j
3Encoder-FeedForwardLayer_2_1/BiasAdd/ReadVariableOp3Encoder-FeedForwardLayer_2_1/BiasAdd/ReadVariableOp2n
5Encoder-FeedForwardLayer_2_1/Tensordot/ReadVariableOp5Encoder-FeedForwardLayer_2_1/Tensordot/ReadVariableOp2�
@Encoder-SelfAttentionLayer-1/attention_output/add/ReadVariableOp@Encoder-SelfAttentionLayer-1/attention_output/add/ReadVariableOp2�
JEncoder-SelfAttentionLayer-1/attention_output/einsum/Einsum/ReadVariableOpJEncoder-SelfAttentionLayer-1/attention_output/einsum/Einsum/ReadVariableOp2j
3Encoder-SelfAttentionLayer-1/key/add/ReadVariableOp3Encoder-SelfAttentionLayer-1/key/add/ReadVariableOp2~
=Encoder-SelfAttentionLayer-1/key/einsum/Einsum/ReadVariableOp=Encoder-SelfAttentionLayer-1/key/einsum/Einsum/ReadVariableOp2n
5Encoder-SelfAttentionLayer-1/query/add/ReadVariableOp5Encoder-SelfAttentionLayer-1/query/add/ReadVariableOp2�
?Encoder-SelfAttentionLayer-1/query/einsum/Einsum/ReadVariableOp?Encoder-SelfAttentionLayer-1/query/einsum/Einsum/ReadVariableOp2n
5Encoder-SelfAttentionLayer-1/value/add/ReadVariableOp5Encoder-SelfAttentionLayer-1/value/add/ReadVariableOp2�
?Encoder-SelfAttentionLayer-1/value/einsum/Einsum/ReadVariableOp?Encoder-SelfAttentionLayer-1/value/einsum/Einsum/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:(
$
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:S O
+
_output_shapes
:���������P	
 
_user_specified_nameinputs
��
�2
"__inference__traced_restore_233910
file_prefix3
%assignvariableop_finallayernorm_gamma:	4
&assignvariableop_1_finallayernorm_beta:	J
8assignvariableop_2_fullyconnectedlayerimprovement_kernel:

D
6assignvariableop_3_fullyconnectedlayerimprovement_bias:
A
/assignvariableop_4_predictionimprovement_kernel:
;
-assignvariableop_5_predictionimprovement_bias:h
Rassignvariableop_6_transformer_encoder_3_encoder_selfattentionlayer_1_query_kernel:	b
Passignvariableop_7_transformer_encoder_3_encoder_selfattentionlayer_1_query_bias:f
Passignvariableop_8_transformer_encoder_3_encoder_selfattentionlayer_1_key_kernel:	`
Nassignvariableop_9_transformer_encoder_3_encoder_selfattentionlayer_1_key_bias:i
Sassignvariableop_10_transformer_encoder_3_encoder_selfattentionlayer_1_value_kernel:	c
Qassignvariableop_11_transformer_encoder_3_encoder_selfattentionlayer_1_value_bias:t
^assignvariableop_12_transformer_encoder_3_encoder_selfattentionlayer_1_attention_output_kernel:	j
\assignvariableop_13_transformer_encoder_3_encoder_selfattentionlayer_1_attention_output_bias:	^
Passignvariableop_14_transformer_encoder_3_encoder_1st_normalizationlayer_1_gamma:	]
Oassignvariableop_15_transformer_encoder_3_encoder_1st_normalizationlayer_1_beta:	^
Passignvariableop_16_transformer_encoder_3_encoder_2nd_normalizationlayer_1_gamma:	]
Oassignvariableop_17_transformer_encoder_3_encoder_2nd_normalizationlayer_1_beta:	_
Massignvariableop_18_transformer_encoder_3_encoder_feedforwardlayer_1_1_kernel:		Y
Kassignvariableop_19_transformer_encoder_3_encoder_feedforwardlayer_1_1_bias:	_
Massignvariableop_20_transformer_encoder_3_encoder_feedforwardlayer_2_1_kernel:		Y
Kassignvariableop_21_transformer_encoder_3_encoder_feedforwardlayer_2_1_bias:	i
Sassignvariableop_22_transformer_encoder_4_encoder_selfattentionlayer_2_query_kernel:	c
Qassignvariableop_23_transformer_encoder_4_encoder_selfattentionlayer_2_query_bias:g
Qassignvariableop_24_transformer_encoder_4_encoder_selfattentionlayer_2_key_kernel:	a
Oassignvariableop_25_transformer_encoder_4_encoder_selfattentionlayer_2_key_bias:i
Sassignvariableop_26_transformer_encoder_4_encoder_selfattentionlayer_2_value_kernel:	c
Qassignvariableop_27_transformer_encoder_4_encoder_selfattentionlayer_2_value_bias:t
^assignvariableop_28_transformer_encoder_4_encoder_selfattentionlayer_2_attention_output_kernel:	j
\assignvariableop_29_transformer_encoder_4_encoder_selfattentionlayer_2_attention_output_bias:	^
Passignvariableop_30_transformer_encoder_4_encoder_1st_normalizationlayer_2_gamma:	]
Oassignvariableop_31_transformer_encoder_4_encoder_1st_normalizationlayer_2_beta:	^
Passignvariableop_32_transformer_encoder_4_encoder_2nd_normalizationlayer_2_gamma:	]
Oassignvariableop_33_transformer_encoder_4_encoder_2nd_normalizationlayer_2_beta:	_
Massignvariableop_34_transformer_encoder_4_encoder_feedforwardlayer_1_2_kernel:		Y
Kassignvariableop_35_transformer_encoder_4_encoder_feedforwardlayer_1_2_bias:	_
Massignvariableop_36_transformer_encoder_4_encoder_feedforwardlayer_2_2_kernel:		Y
Kassignvariableop_37_transformer_encoder_4_encoder_feedforwardlayer_2_2_bias:	i
Sassignvariableop_38_transformer_encoder_5_encoder_selfattentionlayer_3_query_kernel:	c
Qassignvariableop_39_transformer_encoder_5_encoder_selfattentionlayer_3_query_bias:g
Qassignvariableop_40_transformer_encoder_5_encoder_selfattentionlayer_3_key_kernel:	a
Oassignvariableop_41_transformer_encoder_5_encoder_selfattentionlayer_3_key_bias:i
Sassignvariableop_42_transformer_encoder_5_encoder_selfattentionlayer_3_value_kernel:	c
Qassignvariableop_43_transformer_encoder_5_encoder_selfattentionlayer_3_value_bias:t
^assignvariableop_44_transformer_encoder_5_encoder_selfattentionlayer_3_attention_output_kernel:	j
\assignvariableop_45_transformer_encoder_5_encoder_selfattentionlayer_3_attention_output_bias:	^
Passignvariableop_46_transformer_encoder_5_encoder_1st_normalizationlayer_3_gamma:	]
Oassignvariableop_47_transformer_encoder_5_encoder_1st_normalizationlayer_3_beta:	^
Passignvariableop_48_transformer_encoder_5_encoder_2nd_normalizationlayer_3_gamma:	]
Oassignvariableop_49_transformer_encoder_5_encoder_2nd_normalizationlayer_3_beta:	_
Massignvariableop_50_transformer_encoder_5_encoder_feedforwardlayer_1_3_kernel:		Y
Kassignvariableop_51_transformer_encoder_5_encoder_feedforwardlayer_1_3_bias:	_
Massignvariableop_52_transformer_encoder_5_encoder_feedforwardlayer_2_3_kernel:		Y
Kassignvariableop_53_transformer_encoder_5_encoder_feedforwardlayer_2_3_bias:	
identity_55��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_31�AssignVariableOp_32�AssignVariableOp_33�AssignVariableOp_34�AssignVariableOp_35�AssignVariableOp_36�AssignVariableOp_37�AssignVariableOp_38�AssignVariableOp_39�AssignVariableOp_4�AssignVariableOp_40�AssignVariableOp_41�AssignVariableOp_42�AssignVariableOp_43�AssignVariableOp_44�AssignVariableOp_45�AssignVariableOp_46�AssignVariableOp_47�AssignVariableOp_48�AssignVariableOp_49�AssignVariableOp_5�AssignVariableOp_50�AssignVariableOp_51�AssignVariableOp_52�AssignVariableOp_53�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:7*
dtype0*�
value�B�7B5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB'variables/26/.ATTRIBUTES/VARIABLE_VALUEB'variables/27/.ATTRIBUTES/VARIABLE_VALUEB'variables/28/.ATTRIBUTES/VARIABLE_VALUEB'variables/29/.ATTRIBUTES/VARIABLE_VALUEB'variables/30/.ATTRIBUTES/VARIABLE_VALUEB'variables/31/.ATTRIBUTES/VARIABLE_VALUEB'variables/32/.ATTRIBUTES/VARIABLE_VALUEB'variables/33/.ATTRIBUTES/VARIABLE_VALUEB'variables/34/.ATTRIBUTES/VARIABLE_VALUEB'variables/35/.ATTRIBUTES/VARIABLE_VALUEB'variables/36/.ATTRIBUTES/VARIABLE_VALUEB'variables/37/.ATTRIBUTES/VARIABLE_VALUEB'variables/38/.ATTRIBUTES/VARIABLE_VALUEB'variables/39/.ATTRIBUTES/VARIABLE_VALUEB'variables/40/.ATTRIBUTES/VARIABLE_VALUEB'variables/41/.ATTRIBUTES/VARIABLE_VALUEB'variables/42/.ATTRIBUTES/VARIABLE_VALUEB'variables/43/.ATTRIBUTES/VARIABLE_VALUEB'variables/44/.ATTRIBUTES/VARIABLE_VALUEB'variables/45/.ATTRIBUTES/VARIABLE_VALUEB'variables/46/.ATTRIBUTES/VARIABLE_VALUEB'variables/47/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:7*
dtype0*�
valuexBv7B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes�
�:::::::::::::::::::::::::::::::::::::::::::::::::::::::*E
dtypes;
927[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOp%assignvariableop_finallayernorm_gammaIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOp&assignvariableop_1_finallayernorm_betaIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOp8assignvariableop_2_fullyconnectedlayerimprovement_kernelIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOp6assignvariableop_3_fullyconnectedlayerimprovement_biasIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOp/assignvariableop_4_predictionimprovement_kernelIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOp-assignvariableop_5_predictionimprovement_biasIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOpRassignvariableop_6_transformer_encoder_3_encoder_selfattentionlayer_1_query_kernelIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOpPassignvariableop_7_transformer_encoder_3_encoder_selfattentionlayer_1_query_biasIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOpPassignvariableop_8_transformer_encoder_3_encoder_selfattentionlayer_1_key_kernelIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOpNassignvariableop_9_transformer_encoder_3_encoder_selfattentionlayer_1_key_biasIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOpSassignvariableop_10_transformer_encoder_3_encoder_selfattentionlayer_1_value_kernelIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOpQassignvariableop_11_transformer_encoder_3_encoder_selfattentionlayer_1_value_biasIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOp^assignvariableop_12_transformer_encoder_3_encoder_selfattentionlayer_1_attention_output_kernelIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOp\assignvariableop_13_transformer_encoder_3_encoder_selfattentionlayer_1_attention_output_biasIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOpPassignvariableop_14_transformer_encoder_3_encoder_1st_normalizationlayer_1_gammaIdentity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOpOassignvariableop_15_transformer_encoder_3_encoder_1st_normalizationlayer_1_betaIdentity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOpPassignvariableop_16_transformer_encoder_3_encoder_2nd_normalizationlayer_1_gammaIdentity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOpOassignvariableop_17_transformer_encoder_3_encoder_2nd_normalizationlayer_1_betaIdentity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_18AssignVariableOpMassignvariableop_18_transformer_encoder_3_encoder_feedforwardlayer_1_1_kernelIdentity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOpKassignvariableop_19_transformer_encoder_3_encoder_feedforwardlayer_1_1_biasIdentity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_20AssignVariableOpMassignvariableop_20_transformer_encoder_3_encoder_feedforwardlayer_2_1_kernelIdentity_20:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_21AssignVariableOpKassignvariableop_21_transformer_encoder_3_encoder_feedforwardlayer_2_1_biasIdentity_21:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_22AssignVariableOpSassignvariableop_22_transformer_encoder_4_encoder_selfattentionlayer_2_query_kernelIdentity_22:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_23AssignVariableOpQassignvariableop_23_transformer_encoder_4_encoder_selfattentionlayer_2_query_biasIdentity_23:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_24AssignVariableOpQassignvariableop_24_transformer_encoder_4_encoder_selfattentionlayer_2_key_kernelIdentity_24:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_25AssignVariableOpOassignvariableop_25_transformer_encoder_4_encoder_selfattentionlayer_2_key_biasIdentity_25:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_26AssignVariableOpSassignvariableop_26_transformer_encoder_4_encoder_selfattentionlayer_2_value_kernelIdentity_26:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_27AssignVariableOpQassignvariableop_27_transformer_encoder_4_encoder_selfattentionlayer_2_value_biasIdentity_27:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_28AssignVariableOp^assignvariableop_28_transformer_encoder_4_encoder_selfattentionlayer_2_attention_output_kernelIdentity_28:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_29AssignVariableOp\assignvariableop_29_transformer_encoder_4_encoder_selfattentionlayer_2_attention_output_biasIdentity_29:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_30AssignVariableOpPassignvariableop_30_transformer_encoder_4_encoder_1st_normalizationlayer_2_gammaIdentity_30:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_31AssignVariableOpOassignvariableop_31_transformer_encoder_4_encoder_1st_normalizationlayer_2_betaIdentity_31:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_32AssignVariableOpPassignvariableop_32_transformer_encoder_4_encoder_2nd_normalizationlayer_2_gammaIdentity_32:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_33AssignVariableOpOassignvariableop_33_transformer_encoder_4_encoder_2nd_normalizationlayer_2_betaIdentity_33:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_34AssignVariableOpMassignvariableop_34_transformer_encoder_4_encoder_feedforwardlayer_1_2_kernelIdentity_34:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_35AssignVariableOpKassignvariableop_35_transformer_encoder_4_encoder_feedforwardlayer_1_2_biasIdentity_35:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_36AssignVariableOpMassignvariableop_36_transformer_encoder_4_encoder_feedforwardlayer_2_2_kernelIdentity_36:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_37AssignVariableOpKassignvariableop_37_transformer_encoder_4_encoder_feedforwardlayer_2_2_biasIdentity_37:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_38AssignVariableOpSassignvariableop_38_transformer_encoder_5_encoder_selfattentionlayer_3_query_kernelIdentity_38:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_39AssignVariableOpQassignvariableop_39_transformer_encoder_5_encoder_selfattentionlayer_3_query_biasIdentity_39:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_40AssignVariableOpQassignvariableop_40_transformer_encoder_5_encoder_selfattentionlayer_3_key_kernelIdentity_40:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_41AssignVariableOpOassignvariableop_41_transformer_encoder_5_encoder_selfattentionlayer_3_key_biasIdentity_41:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_42AssignVariableOpSassignvariableop_42_transformer_encoder_5_encoder_selfattentionlayer_3_value_kernelIdentity_42:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_43AssignVariableOpQassignvariableop_43_transformer_encoder_5_encoder_selfattentionlayer_3_value_biasIdentity_43:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_44AssignVariableOp^assignvariableop_44_transformer_encoder_5_encoder_selfattentionlayer_3_attention_output_kernelIdentity_44:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_45AssignVariableOp\assignvariableop_45_transformer_encoder_5_encoder_selfattentionlayer_3_attention_output_biasIdentity_45:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_46AssignVariableOpPassignvariableop_46_transformer_encoder_5_encoder_1st_normalizationlayer_3_gammaIdentity_46:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_47AssignVariableOpOassignvariableop_47_transformer_encoder_5_encoder_1st_normalizationlayer_3_betaIdentity_47:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_48AssignVariableOpPassignvariableop_48_transformer_encoder_5_encoder_2nd_normalizationlayer_3_gammaIdentity_48:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_49AssignVariableOpOassignvariableop_49_transformer_encoder_5_encoder_2nd_normalizationlayer_3_betaIdentity_49:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_50AssignVariableOpMassignvariableop_50_transformer_encoder_5_encoder_feedforwardlayer_1_3_kernelIdentity_50:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_51AssignVariableOpKassignvariableop_51_transformer_encoder_5_encoder_feedforwardlayer_1_3_biasIdentity_51:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_52AssignVariableOpMassignvariableop_52_transformer_encoder_5_encoder_feedforwardlayer_2_3_kernelIdentity_52:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_53AssignVariableOpKassignvariableop_53_transformer_encoder_5_encoder_feedforwardlayer_2_3_biasIdentity_53:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 �	
Identity_54Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_55IdentityIdentity_54:output:0^NoOp_1*
T0*
_output_shapes
: �	
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
_output_shapes
 "#
identity_55Identity_55:output:0*(
_construction_contextkEagerRuntime*�
_input_shapesp
n: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92$
AssignVariableOpAssignVariableOp:W6S
Q
_user_specified_name97transformer_encoder_5/Encoder-FeedForwardLayer_2_3/bias:Y5U
S
_user_specified_name;9transformer_encoder_5/Encoder-FeedForwardLayer_2_3/kernel:W4S
Q
_user_specified_name97transformer_encoder_5/Encoder-FeedForwardLayer_1_3/bias:Y3U
S
_user_specified_name;9transformer_encoder_5/Encoder-FeedForwardLayer_1_3/kernel:[2W
U
_user_specified_name=;transformer_encoder_5/Encoder-2nd-NormalizationLayer-3/beta:\1X
V
_user_specified_name><transformer_encoder_5/Encoder-2nd-NormalizationLayer-3/gamma:[0W
U
_user_specified_name=;transformer_encoder_5/Encoder-1st-NormalizationLayer-3/beta:\/X
V
_user_specified_name><transformer_encoder_5/Encoder-1st-NormalizationLayer-3/gamma:h.d
b
_user_specified_nameJHtransformer_encoder_5/Encoder-SelfAttentionLayer-3/attention_output/bias:j-f
d
_user_specified_nameLJtransformer_encoder_5/Encoder-SelfAttentionLayer-3/attention_output/kernel:],Y
W
_user_specified_name?=transformer_encoder_5/Encoder-SelfAttentionLayer-3/value/bias:_+[
Y
_user_specified_nameA?transformer_encoder_5/Encoder-SelfAttentionLayer-3/value/kernel:[*W
U
_user_specified_name=;transformer_encoder_5/Encoder-SelfAttentionLayer-3/key/bias:])Y
W
_user_specified_name?=transformer_encoder_5/Encoder-SelfAttentionLayer-3/key/kernel:](Y
W
_user_specified_name?=transformer_encoder_5/Encoder-SelfAttentionLayer-3/query/bias:_'[
Y
_user_specified_nameA?transformer_encoder_5/Encoder-SelfAttentionLayer-3/query/kernel:W&S
Q
_user_specified_name97transformer_encoder_4/Encoder-FeedForwardLayer_2_2/bias:Y%U
S
_user_specified_name;9transformer_encoder_4/Encoder-FeedForwardLayer_2_2/kernel:W$S
Q
_user_specified_name97transformer_encoder_4/Encoder-FeedForwardLayer_1_2/bias:Y#U
S
_user_specified_name;9transformer_encoder_4/Encoder-FeedForwardLayer_1_2/kernel:["W
U
_user_specified_name=;transformer_encoder_4/Encoder-2nd-NormalizationLayer-2/beta:\!X
V
_user_specified_name><transformer_encoder_4/Encoder-2nd-NormalizationLayer-2/gamma:[ W
U
_user_specified_name=;transformer_encoder_4/Encoder-1st-NormalizationLayer-2/beta:\X
V
_user_specified_name><transformer_encoder_4/Encoder-1st-NormalizationLayer-2/gamma:hd
b
_user_specified_nameJHtransformer_encoder_4/Encoder-SelfAttentionLayer-2/attention_output/bias:jf
d
_user_specified_nameLJtransformer_encoder_4/Encoder-SelfAttentionLayer-2/attention_output/kernel:]Y
W
_user_specified_name?=transformer_encoder_4/Encoder-SelfAttentionLayer-2/value/bias:_[
Y
_user_specified_nameA?transformer_encoder_4/Encoder-SelfAttentionLayer-2/value/kernel:[W
U
_user_specified_name=;transformer_encoder_4/Encoder-SelfAttentionLayer-2/key/bias:]Y
W
_user_specified_name?=transformer_encoder_4/Encoder-SelfAttentionLayer-2/key/kernel:]Y
W
_user_specified_name?=transformer_encoder_4/Encoder-SelfAttentionLayer-2/query/bias:_[
Y
_user_specified_nameA?transformer_encoder_4/Encoder-SelfAttentionLayer-2/query/kernel:WS
Q
_user_specified_name97transformer_encoder_3/Encoder-FeedForwardLayer_2_1/bias:YU
S
_user_specified_name;9transformer_encoder_3/Encoder-FeedForwardLayer_2_1/kernel:WS
Q
_user_specified_name97transformer_encoder_3/Encoder-FeedForwardLayer_1_1/bias:YU
S
_user_specified_name;9transformer_encoder_3/Encoder-FeedForwardLayer_1_1/kernel:[W
U
_user_specified_name=;transformer_encoder_3/Encoder-2nd-NormalizationLayer-1/beta:\X
V
_user_specified_name><transformer_encoder_3/Encoder-2nd-NormalizationLayer-1/gamma:[W
U
_user_specified_name=;transformer_encoder_3/Encoder-1st-NormalizationLayer-1/beta:\X
V
_user_specified_name><transformer_encoder_3/Encoder-1st-NormalizationLayer-1/gamma:hd
b
_user_specified_nameJHtransformer_encoder_3/Encoder-SelfAttentionLayer-1/attention_output/bias:jf
d
_user_specified_nameLJtransformer_encoder_3/Encoder-SelfAttentionLayer-1/attention_output/kernel:]Y
W
_user_specified_name?=transformer_encoder_3/Encoder-SelfAttentionLayer-1/value/bias:_[
Y
_user_specified_nameA?transformer_encoder_3/Encoder-SelfAttentionLayer-1/value/kernel:[
W
U
_user_specified_name=;transformer_encoder_3/Encoder-SelfAttentionLayer-1/key/bias:]	Y
W
_user_specified_name?=transformer_encoder_3/Encoder-SelfAttentionLayer-1/key/kernel:]Y
W
_user_specified_name?=transformer_encoder_3/Encoder-SelfAttentionLayer-1/query/bias:_[
Y
_user_specified_nameA?transformer_encoder_3/Encoder-SelfAttentionLayer-1/query/kernel::6
4
_user_specified_namePredictionImprovement/bias:<8
6
_user_specified_namePredictionImprovement/kernel:C?
=
_user_specified_name%#FullyConnectedLayerImprovement/bias:EA
?
_user_specified_name'%FullyConnectedLayerImprovement/kernel:3/
-
_user_specified_nameFinalLayerNorm/beta:40
.
_user_specified_nameFinalLayerNorm/gamma:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
]
A__inference_ReduceStackDimensionViaSummation_layer_call_fn_233265

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *e
f`R^
\__inference_ReduceStackDimensionViaSummation_layer_call_and_return_conditional_losses_230490`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������	"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������P	:S O
+
_output_shapes
:���������P	
 
_user_specified_nameinputs
� 
�
J__inference_FinalLayerNorm_layer_call_and_return_conditional_losses_230477

inputs)
mul_readvariableop_resource:	)
add_readvariableop_resource:	
identity��add/ReadVariableOp�mul/ReadVariableOpI
ShapeShapeinputs*
T0*
_output_shapes
::��]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskO
ConstConst*
_output_shapes
:*
dtype0*
valueB: U
ProdProdstrided_slice:output:0Const:output:0*
T0*
_output_shapes
: _
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_maskQ
Const_1Const*
_output_shapes
:*
dtype0*
valueB: [
Prod_1Prodstrided_slice_1:output:0Const_1:output:0*
T0*
_output_shapes
: Q
Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :Q
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :�
Reshape/shapePackReshape/shape/0:output:0Prod:output:0Prod_1:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:u
ReshapeReshapeinputsReshape/shape:output:0*
T0*8
_output_shapes&
$:"������������������P
ones/packedPackProd:output:0*
N*
T0*
_output_shapes
:O

ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?e
onesFillones/packed:output:0ones/Const:output:0*
T0*#
_output_shapes
:���������Q
zeros/packedPackProd:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    h
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*#
_output_shapes
:���������J
Const_2Const*
_output_shapes
: *
dtype0*
valueB J
Const_3Const*
_output_shapes
: *
dtype0*
valueB �
FusedBatchNormV3FusedBatchNormV3Reshape:output:0ones:output:0zeros:output:0Const_2:output:0Const_3:output:0*
T0*
U0*x
_output_shapesf
d:"������������������:���������:���������:���������:���������:*
data_formatNCHW*
epsilon%o�:p
	Reshape_1ReshapeFusedBatchNormV3:y:0Shape:output:0*
T0*+
_output_shapes
:���������P	j
mul/ReadVariableOpReadVariableOpmul_readvariableop_resource*
_output_shapes
:	*
dtype0p
mulMulReshape_1:output:0mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������P	j
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
:	*
dtype0g
addAddV2mul:z:0add/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������P	Z
IdentityIdentityadd:z:0^NoOp*
T0*+
_output_shapes
:���������P	L
NoOpNoOp^add/ReadVariableOp^mul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������P	: : 2(
add/ReadVariableOpadd/ReadVariableOp2(
mul/ReadVariableOpmul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:S O
+
_output_shapes
:���������P	
 
_user_specified_nameinputs
�
l
P__inference_StandardizeTimeLimit_layer_call_and_return_conditional_losses_231208

inputs
identityJ
sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  pAT
subSubinputssub/y:output:0*
T0*'
_output_shapes
:���������N
	truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �@a
truedivRealDivsub:z:0truediv/y:output:0*
T0*'
_output_shapes
:���������S
IdentityIdentitytruediv:z:0*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
6__inference_transformer_encoder_4_layer_call_fn_232368

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:
	unknown_3:	
	unknown_4:
	unknown_5:	
	unknown_6:
	unknown_7:	
	unknown_8:	
	unknown_9:	

unknown_10:	

unknown_11:		

unknown_12:	

unknown_13:		

unknown_14:	
identity

identity_1��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *F
_output_shapes4
2:���������P	:���������PP*2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *Z
fURS
Q__inference_transformer_encoder_4_layer_call_and_return_conditional_losses_230179s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������P	y

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*/
_output_shapes
:���������PP<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:���������P	: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name232362:&"
 
_user_specified_name232360:&"
 
_user_specified_name232358:&"
 
_user_specified_name232356:&"
 
_user_specified_name232354:&"
 
_user_specified_name232352:&
"
 
_user_specified_name232350:&	"
 
_user_specified_name232348:&"
 
_user_specified_name232346:&"
 
_user_specified_name232344:&"
 
_user_specified_name232342:&"
 
_user_specified_name232340:&"
 
_user_specified_name232338:&"
 
_user_specified_name232336:&"
 
_user_specified_name232334:&"
 
_user_specified_name232332:S O
+
_output_shapes
:���������P	
 
_user_specified_nameinputs
�
�
6__inference_transformer_encoder_3_layer_call_fn_231967

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:
	unknown_3:	
	unknown_4:
	unknown_5:	
	unknown_6:
	unknown_7:	
	unknown_8:	
	unknown_9:	

unknown_10:	

unknown_11:		

unknown_12:	

unknown_13:		

unknown_14:	
identity

identity_1��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *F
_output_shapes4
2:���������P	:���������PP*2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *Z
fURS
Q__inference_transformer_encoder_3_layer_call_and_return_conditional_losses_230735s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������P	y

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*/
_output_shapes
:���������PP<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:���������P	: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name231961:&"
 
_user_specified_name231959:&"
 
_user_specified_name231957:&"
 
_user_specified_name231955:&"
 
_user_specified_name231953:&"
 
_user_specified_name231951:&
"
 
_user_specified_name231949:&	"
 
_user_specified_name231947:&"
 
_user_specified_name231945:&"
 
_user_specified_name231943:&"
 
_user_specified_name231941:&"
 
_user_specified_name231939:&"
 
_user_specified_name231937:&"
 
_user_specified_name231935:&"
 
_user_specified_name231933:&"
 
_user_specified_name231931:S O
+
_output_shapes
:���������P	
 
_user_specified_nameinputs
�
]
A__inference_ReduceStackDimensionViaSummation_layer_call_fn_233270

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *e
f`R^
\__inference_ReduceStackDimensionViaSummation_layer_call_and_return_conditional_losses_231198`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������	"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������P	:S O
+
_output_shapes
:���������P	
 
_user_specified_nameinputs
�
C
'__inference_Output_layer_call_fn_233382

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_Output_layer_call_and_return_conditional_losses_231225Q
IdentityIdentityPartitionedCall:output:0*
T0*
_output_shapes
:"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
?__inference_FullyConnectedLayerImprovement_layer_call_fn_233334

inputs
unknown:


	unknown_0:

identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *c
f^R\
Z__inference_FullyConnectedLayerImprovement_layer_call_and_return_conditional_losses_230527o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������
<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������
: : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name233330:&"
 
_user_specified_name233328:O K
'
_output_shapes
:���������

 
_user_specified_nameinputs"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
_
StackLevelInputFeaturesD
)serving_default_StackLevelInputFeatures:0���������P	
I
TimeLimitInput7
 serving_default_TimeLimitInput:0���������+
Output!
StatefulPartitionedCall:0tensorflow/serving/predict:��
�
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer-6
layer-7
	layer-8

layer-9
layer_with_weights-4
layer-10
layer_with_weights-5
layer-11
layer-12
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures"
_tf_keras_network
"
_tf_keras_input_layer
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
�
	variables
trainable_variables
regularization_losses
	keras_api
 __call__
*!&call_and_return_all_conditional_losses
"self_attention_layer
#add1
$add2
%
layernorm1
&
layernorm2
'feed_forward_layer_1
(feed_forward_layer_2
)dropout_layer"
_tf_keras_layer
�
*	variables
+trainable_variables
,regularization_losses
-	keras_api
.__call__
*/&call_and_return_all_conditional_losses
0self_attention_layer
1add1
2add2
3
layernorm1
4
layernorm2
5feed_forward_layer_1
6feed_forward_layer_2
7dropout_layer"
_tf_keras_layer
�
8	variables
9trainable_variables
:regularization_losses
;	keras_api
<__call__
*=&call_and_return_all_conditional_losses
>self_attention_layer
?add1
@add2
A
layernorm1
B
layernorm2
Cfeed_forward_layer_1
Dfeed_forward_layer_2
Edropout_layer"
_tf_keras_layer
�
F	variables
Gtrainable_variables
Hregularization_losses
I	keras_api
J__call__
*K&call_and_return_all_conditional_losses
Laxis
	Mgamma
Nbeta"
_tf_keras_layer
"
_tf_keras_input_layer
�
O	variables
Ptrainable_variables
Qregularization_losses
R	keras_api
S__call__
*T&call_and_return_all_conditional_losses"
_tf_keras_layer
�
U	variables
Vtrainable_variables
Wregularization_losses
X	keras_api
Y__call__
*Z&call_and_return_all_conditional_losses"
_tf_keras_layer
�
[	variables
\trainable_variables
]regularization_losses
^	keras_api
___call__
*`&call_and_return_all_conditional_losses"
_tf_keras_layer
�
a	variables
btrainable_variables
cregularization_losses
d	keras_api
e__call__
*f&call_and_return_all_conditional_losses

gkernel
hbias"
_tf_keras_layer
�
i	variables
jtrainable_variables
kregularization_losses
l	keras_api
m__call__
*n&call_and_return_all_conditional_losses

okernel
pbias"
_tf_keras_layer
�
q	variables
rtrainable_variables
sregularization_losses
t	keras_api
u__call__
*v&call_and_return_all_conditional_losses"
_tf_keras_layer
�
w0
x1
y2
z3
{4
|5
}6
~7
8
�9
�10
�11
�12
�13
�14
�15
�16
�17
�18
�19
�20
�21
�22
�23
�24
�25
�26
�27
�28
�29
�30
�31
�32
�33
�34
�35
�36
�37
�38
�39
�40
�41
�42
�43
�44
�45
�46
�47
M48
N49
g50
h51
o52
p53"
trackable_list_wrapper
�
w0
x1
y2
z3
{4
|5
}6
~7
8
�9
�10
�11
�12
�13
�14
�15
�16
�17
�18
�19
�20
�21
�22
�23
�24
�25
�26
�27
�28
�29
�30
�31
�32
�33
�34
�35
�36
�37
�38
�39
�40
�41
�42
�43
�44
�45
�46
�47
M48
N49
g50
h51
o52
p53"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
(__inference_model_1_layer_call_fn_231342
(__inference_model_1_layer_call_fn_231456�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
C__inference_model_1_layer_call_and_return_conditional_losses_230556
C__inference_model_1_layer_call_and_return_conditional_losses_231228�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�B�
!__inference__wrapped_model_229753StackLevelInputFeaturesTimeLimitInput"�
���
FullArgSpec
args�

jargs_0
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
-
�serving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
-__inference_MaskingLayer_layer_call_fn_231878�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
H__inference_MaskingLayer_layer_call_and_return_conditional_losses_231889�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
w0
x1
y2
z3
{4
|5
}6
~7
8
�9
�10
�11
�12
�13
�14
�15"
trackable_list_wrapper
�
w0
x1
y2
z3
{4
|5
}6
~7
8
�9
�10
�11
�12
�13
�14
�15"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
 __call__
*!&call_and_return_all_conditional_losses
&!"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
6__inference_transformer_encoder_3_layer_call_fn_231928
6__inference_transformer_encoder_3_layer_call_fn_231967�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
Q__inference_transformer_encoder_3_layer_call_and_return_conditional_losses_232155
Q__inference_transformer_encoder_3_layer_call_and_return_conditional_losses_232329�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 z�trace_0z�trace_1
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�_query_dense
�
_key_dense
�_value_dense
�_softmax
�_dropout_layer
�_output_dense"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis
	gamma
	�beta"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis

�gamma
	�beta"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�_random_generator"
_tf_keras_layer
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15"
trackable_list_wrapper
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
*	variables
+trainable_variables
,regularization_losses
.__call__
*/&call_and_return_all_conditional_losses
&/"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
6__inference_transformer_encoder_4_layer_call_fn_232368
6__inference_transformer_encoder_4_layer_call_fn_232407�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
Q__inference_transformer_encoder_4_layer_call_and_return_conditional_losses_232595
Q__inference_transformer_encoder_4_layer_call_and_return_conditional_losses_232769�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 z�trace_0z�trace_1
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�_query_dense
�
_key_dense
�_value_dense
�_softmax
�_dropout_layer
�_output_dense"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis

�gamma
	�beta"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis

�gamma
	�beta"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�_random_generator"
_tf_keras_layer
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15"
trackable_list_wrapper
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
8	variables
9trainable_variables
:regularization_losses
<__call__
*=&call_and_return_all_conditional_losses
&="call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
6__inference_transformer_encoder_5_layer_call_fn_232808
6__inference_transformer_encoder_5_layer_call_fn_232847�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
Q__inference_transformer_encoder_5_layer_call_and_return_conditional_losses_233035
Q__inference_transformer_encoder_5_layer_call_and_return_conditional_losses_233209�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 z�trace_0z�trace_1
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�_query_dense
�
_key_dense
�_value_dense
�_softmax
�_dropout_layer
�_output_dense"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis

�gamma
	�beta"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis

�gamma
	�beta"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�_random_generator"
_tf_keras_layer
.
M0
N1"
trackable_list_wrapper
.
M0
N1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
F	variables
Gtrainable_variables
Hregularization_losses
J__call__
*K&call_and_return_all_conditional_losses
&K"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
/__inference_FinalLayerNorm_layer_call_fn_233218�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
J__inference_FinalLayerNorm_layer_call_and_return_conditional_losses_233260�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
 "
trackable_list_wrapper
": 	2FinalLayerNorm/gamma
!:	2FinalLayerNorm/beta
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
O	variables
Ptrainable_variables
Qregularization_losses
S__call__
*T&call_and_return_all_conditional_losses
&T"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
A__inference_ReduceStackDimensionViaSummation_layer_call_fn_233265
A__inference_ReduceStackDimensionViaSummation_layer_call_fn_233270�
���
FullArgSpec)
args!�
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults�

 
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
\__inference_ReduceStackDimensionViaSummation_layer_call_and_return_conditional_losses_233278
\__inference_ReduceStackDimensionViaSummation_layer_call_and_return_conditional_losses_233286�
���
FullArgSpec)
args!�
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults�

 
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
U	variables
Vtrainable_variables
Wregularization_losses
Y__call__
*Z&call_and_return_all_conditional_losses
&Z"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
5__inference_StandardizeTimeLimit_layer_call_fn_233291
5__inference_StandardizeTimeLimit_layer_call_fn_233296�
���
FullArgSpec)
args!�
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults�

 
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
P__inference_StandardizeTimeLimit_layer_call_and_return_conditional_losses_233304
P__inference_StandardizeTimeLimit_layer_call_and_return_conditional_losses_233312�
���
FullArgSpec)
args!�
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults�

 
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
[	variables
\trainable_variables
]regularization_losses
___call__
*`&call_and_return_all_conditional_losses
&`"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
1__inference_ConcatenateLayer_layer_call_fn_233318�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
L__inference_ConcatenateLayer_layer_call_and_return_conditional_losses_233325�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
.
g0
h1"
trackable_list_wrapper
.
g0
h1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
a	variables
btrainable_variables
cregularization_losses
e__call__
*f&call_and_return_all_conditional_losses
&f"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
?__inference_FullyConnectedLayerImprovement_layer_call_fn_233334�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
Z__inference_FullyConnectedLayerImprovement_layer_call_and_return_conditional_losses_233352�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
7:5

2%FullyConnectedLayerImprovement/kernel
1:/
2#FullyConnectedLayerImprovement/bias
.
o0
p1"
trackable_list_wrapper
.
o0
p1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
i	variables
jtrainable_variables
kregularization_losses
m__call__
*n&call_and_return_all_conditional_losses
&n"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
6__inference_PredictionImprovement_layer_call_fn_233361�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
Q__inference_PredictionImprovement_layer_call_and_return_conditional_losses_233372�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
.:,
2PredictionImprovement/kernel
(:&2PredictionImprovement/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
q	variables
rtrainable_variables
sregularization_losses
u__call__
*v&call_and_return_all_conditional_losses
&v"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
'__inference_Output_layer_call_fn_233377
'__inference_Output_layer_call_fn_233382�
���
FullArgSpec)
args!�
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults�

 
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
B__inference_Output_layer_call_and_return_conditional_losses_233387
B__inference_Output_layer_call_and_return_conditional_losses_233392�
���
FullArgSpec)
args!�
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults�

 
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
U:S	2?transformer_encoder_3/Encoder-SelfAttentionLayer-1/query/kernel
O:M2=transformer_encoder_3/Encoder-SelfAttentionLayer-1/query/bias
S:Q	2=transformer_encoder_3/Encoder-SelfAttentionLayer-1/key/kernel
M:K2;transformer_encoder_3/Encoder-SelfAttentionLayer-1/key/bias
U:S	2?transformer_encoder_3/Encoder-SelfAttentionLayer-1/value/kernel
O:M2=transformer_encoder_3/Encoder-SelfAttentionLayer-1/value/bias
`:^	2Jtransformer_encoder_3/Encoder-SelfAttentionLayer-1/attention_output/kernel
V:T	2Htransformer_encoder_3/Encoder-SelfAttentionLayer-1/attention_output/bias
J:H	2<transformer_encoder_3/Encoder-1st-NormalizationLayer-1/gamma
I:G	2;transformer_encoder_3/Encoder-1st-NormalizationLayer-1/beta
J:H	2<transformer_encoder_3/Encoder-2nd-NormalizationLayer-1/gamma
I:G	2;transformer_encoder_3/Encoder-2nd-NormalizationLayer-1/beta
K:I		29transformer_encoder_3/Encoder-FeedForwardLayer_1_1/kernel
E:C	27transformer_encoder_3/Encoder-FeedForwardLayer_1_1/bias
K:I		29transformer_encoder_3/Encoder-FeedForwardLayer_2_1/kernel
E:C	27transformer_encoder_3/Encoder-FeedForwardLayer_2_1/bias
U:S	2?transformer_encoder_4/Encoder-SelfAttentionLayer-2/query/kernel
O:M2=transformer_encoder_4/Encoder-SelfAttentionLayer-2/query/bias
S:Q	2=transformer_encoder_4/Encoder-SelfAttentionLayer-2/key/kernel
M:K2;transformer_encoder_4/Encoder-SelfAttentionLayer-2/key/bias
U:S	2?transformer_encoder_4/Encoder-SelfAttentionLayer-2/value/kernel
O:M2=transformer_encoder_4/Encoder-SelfAttentionLayer-2/value/bias
`:^	2Jtransformer_encoder_4/Encoder-SelfAttentionLayer-2/attention_output/kernel
V:T	2Htransformer_encoder_4/Encoder-SelfAttentionLayer-2/attention_output/bias
J:H	2<transformer_encoder_4/Encoder-1st-NormalizationLayer-2/gamma
I:G	2;transformer_encoder_4/Encoder-1st-NormalizationLayer-2/beta
J:H	2<transformer_encoder_4/Encoder-2nd-NormalizationLayer-2/gamma
I:G	2;transformer_encoder_4/Encoder-2nd-NormalizationLayer-2/beta
K:I		29transformer_encoder_4/Encoder-FeedForwardLayer_1_2/kernel
E:C	27transformer_encoder_4/Encoder-FeedForwardLayer_1_2/bias
K:I		29transformer_encoder_4/Encoder-FeedForwardLayer_2_2/kernel
E:C	27transformer_encoder_4/Encoder-FeedForwardLayer_2_2/bias
U:S	2?transformer_encoder_5/Encoder-SelfAttentionLayer-3/query/kernel
O:M2=transformer_encoder_5/Encoder-SelfAttentionLayer-3/query/bias
S:Q	2=transformer_encoder_5/Encoder-SelfAttentionLayer-3/key/kernel
M:K2;transformer_encoder_5/Encoder-SelfAttentionLayer-3/key/bias
U:S	2?transformer_encoder_5/Encoder-SelfAttentionLayer-3/value/kernel
O:M2=transformer_encoder_5/Encoder-SelfAttentionLayer-3/value/bias
`:^	2Jtransformer_encoder_5/Encoder-SelfAttentionLayer-3/attention_output/kernel
V:T	2Htransformer_encoder_5/Encoder-SelfAttentionLayer-3/attention_output/bias
J:H	2<transformer_encoder_5/Encoder-1st-NormalizationLayer-3/gamma
I:G	2;transformer_encoder_5/Encoder-1st-NormalizationLayer-3/beta
J:H	2<transformer_encoder_5/Encoder-2nd-NormalizationLayer-3/gamma
I:G	2;transformer_encoder_5/Encoder-2nd-NormalizationLayer-3/beta
K:I		29transformer_encoder_5/Encoder-FeedForwardLayer_1_3/kernel
E:C	27transformer_encoder_5/Encoder-FeedForwardLayer_1_3/bias
K:I		29transformer_encoder_5/Encoder-FeedForwardLayer_2_3/kernel
E:C	27transformer_encoder_5/Encoder-FeedForwardLayer_2_3/bias
 "
trackable_list_wrapper
~
0
1
2
3
4
5
6
7
	8

9
10
11
12"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
(__inference_model_1_layer_call_fn_231342StackLevelInputFeaturesTimeLimitInput"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
(__inference_model_1_layer_call_fn_231456StackLevelInputFeaturesTimeLimitInput"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_model_1_layer_call_and_return_conditional_losses_230556StackLevelInputFeaturesTimeLimitInput"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_model_1_layer_call_and_return_conditional_losses_231228StackLevelInputFeaturesTimeLimitInput"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
$__inference_signature_wrapper_231873StackLevelInputFeaturesTimeLimitInput"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 >

kwonlyargs0�-
jStackLevelInputFeatures
jTimeLimitInput
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
-__inference_MaskingLayer_layer_call_fn_231878inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
H__inference_MaskingLayer_layer_call_and_return_conditional_losses_231889inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
X
"0
#1
$2
%3
&4
'5
(6
)7"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
6__inference_transformer_encoder_3_layer_call_fn_231928inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs�

jtraining
kwonlydefaults
 
annotations� *
 
�B�
6__inference_transformer_encoder_3_layer_call_fn_231967inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs�

jtraining
kwonlydefaults
 
annotations� *
 
�B�
Q__inference_transformer_encoder_3_layer_call_and_return_conditional_losses_232155inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs�

jtraining
kwonlydefaults
 
annotations� *
 
�B�
Q__inference_transformer_encoder_3_layer_call_and_return_conditional_losses_232329inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs�

jtraining
kwonlydefaults
 
annotations� *
 
X
w0
x1
y2
z3
{4
|5
}6
~7"
trackable_list_wrapper
X
w0
x1
y2
z3
{4
|5
}6
~7"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpecp
argsh�e
jquery
jvalue
jkey
jattention_mask
jreturn_attention_scores

jtraining
juse_causal_mask
varargs
 
varkw
 #
defaults�

 

 
p 
p 
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpecp
argsh�e
jquery
jvalue
jkey
jattention_mask
jreturn_attention_scores

jtraining
juse_causal_mask
varargs
 
varkw
 #
defaults�

 

 
p 
p 
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�partial_output_shape
�full_output_shape

wkernel
xbias"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�partial_output_shape
�full_output_shape

ykernel
zbias"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�partial_output_shape
�full_output_shape

{kernel
|bias"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�_random_generator"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�partial_output_shape
�full_output_shape

}kernel
~bias"
_tf_keras_layer
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
/
0
�1"
trackable_list_wrapper
/
0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
"
_generic_user_object
 "
trackable_list_wrapper
X
00
11
22
33
44
55
66
77"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
6__inference_transformer_encoder_4_layer_call_fn_232368inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs�

jtraining
kwonlydefaults
 
annotations� *
 
�B�
6__inference_transformer_encoder_4_layer_call_fn_232407inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs�

jtraining
kwonlydefaults
 
annotations� *
 
�B�
Q__inference_transformer_encoder_4_layer_call_and_return_conditional_losses_232595inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs�

jtraining
kwonlydefaults
 
annotations� *
 
�B�
Q__inference_transformer_encoder_4_layer_call_and_return_conditional_losses_232769inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs�

jtraining
kwonlydefaults
 
annotations� *
 
`
�0
�1
�2
�3
�4
�5
�6
�7"
trackable_list_wrapper
`
�0
�1
�2
�3
�4
�5
�6
�7"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpecp
argsh�e
jquery
jvalue
jkey
jattention_mask
jreturn_attention_scores

jtraining
juse_causal_mask
varargs
 
varkw
 #
defaults�

 

 
p 
p 
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpecp
argsh�e
jquery
jvalue
jkey
jattention_mask
jreturn_attention_scores

jtraining
juse_causal_mask
varargs
 
varkw
 #
defaults�

 

 
p 
p 
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�partial_output_shape
�full_output_shape
�kernel
	�bias"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�partial_output_shape
�full_output_shape
�kernel
	�bias"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�partial_output_shape
�full_output_shape
�kernel
	�bias"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�_random_generator"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�partial_output_shape
�full_output_shape
�kernel
	�bias"
_tf_keras_layer
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
"
_generic_user_object
 "
trackable_list_wrapper
X
>0
?1
@2
A3
B4
C5
D6
E7"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
6__inference_transformer_encoder_5_layer_call_fn_232808inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs�

jtraining
kwonlydefaults
 
annotations� *
 
�B�
6__inference_transformer_encoder_5_layer_call_fn_232847inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs�

jtraining
kwonlydefaults
 
annotations� *
 
�B�
Q__inference_transformer_encoder_5_layer_call_and_return_conditional_losses_233035inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs�

jtraining
kwonlydefaults
 
annotations� *
 
�B�
Q__inference_transformer_encoder_5_layer_call_and_return_conditional_losses_233209inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs�

jtraining
kwonlydefaults
 
annotations� *
 
`
�0
�1
�2
�3
�4
�5
�6
�7"
trackable_list_wrapper
`
�0
�1
�2
�3
�4
�5
�6
�7"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpecp
argsh�e
jquery
jvalue
jkey
jattention_mask
jreturn_attention_scores

jtraining
juse_causal_mask
varargs
 
varkw
 #
defaults�

 

 
p 
p 
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpecp
argsh�e
jquery
jvalue
jkey
jattention_mask
jreturn_attention_scores

jtraining
juse_causal_mask
varargs
 
varkw
 #
defaults�

 

 
p 
p 
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�partial_output_shape
�full_output_shape
�kernel
	�bias"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�partial_output_shape
�full_output_shape
�kernel
	�bias"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�partial_output_shape
�full_output_shape
�kernel
	�bias"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�_random_generator"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�partial_output_shape
�full_output_shape
�kernel
	�bias"
_tf_keras_layer
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
/__inference_FinalLayerNorm_layer_call_fn_233218inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
J__inference_FinalLayerNorm_layer_call_and_return_conditional_losses_233260inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
A__inference_ReduceStackDimensionViaSummation_layer_call_fn_233265inputs"�
���
FullArgSpec)
args!�
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
A__inference_ReduceStackDimensionViaSummation_layer_call_fn_233270inputs"�
���
FullArgSpec)
args!�
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
\__inference_ReduceStackDimensionViaSummation_layer_call_and_return_conditional_losses_233278inputs"�
���
FullArgSpec)
args!�
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
\__inference_ReduceStackDimensionViaSummation_layer_call_and_return_conditional_losses_233286inputs"�
���
FullArgSpec)
args!�
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
5__inference_StandardizeTimeLimit_layer_call_fn_233291inputs"�
���
FullArgSpec)
args!�
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
5__inference_StandardizeTimeLimit_layer_call_fn_233296inputs"�
���
FullArgSpec)
args!�
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
P__inference_StandardizeTimeLimit_layer_call_and_return_conditional_losses_233304inputs"�
���
FullArgSpec)
args!�
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
P__inference_StandardizeTimeLimit_layer_call_and_return_conditional_losses_233312inputs"�
���
FullArgSpec)
args!�
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
1__inference_ConcatenateLayer_layer_call_fn_233318inputs_0inputs_1"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
L__inference_ConcatenateLayer_layer_call_and_return_conditional_losses_233325inputs_0inputs_1"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
?__inference_FullyConnectedLayerImprovement_layer_call_fn_233334inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
Z__inference_FullyConnectedLayerImprovement_layer_call_and_return_conditional_losses_233352inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
6__inference_PredictionImprovement_layer_call_fn_233361inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
Q__inference_PredictionImprovement_layer_call_and_return_conditional_losses_233372inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
'__inference_Output_layer_call_fn_233377inputs"�
���
FullArgSpec)
args!�
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
'__inference_Output_layer_call_fn_233382inputs"�
���
FullArgSpec)
args!�
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
B__inference_Output_layer_call_and_return_conditional_losses_233387inputs"�
���
FullArgSpec)
args!�
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
B__inference_Output_layer_call_and_return_conditional_losses_233392inputs"�
���
FullArgSpec)
args!�
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
P
�0
�1
�2
�3
�4
�5"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
w0
x1"
trackable_list_wrapper
.
w0
x1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
y0
z1"
trackable_list_wrapper
.
y0
z1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
{0
|1"
trackable_list_wrapper
.
{0
|1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�
jinputs
jmask
varargs
 
varkw
 
defaults�

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jinputs
jmask
varargs
 
varkw
 
defaults�

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
"
_generic_user_object
.
}0
~1"
trackable_list_wrapper
.
}0
~1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
P
�0
�1
�2
�3
�4
�5"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�
jinputs
jmask
varargs
 
varkw
 
defaults�

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jinputs
jmask
varargs
 
varkw
 
defaults�

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
"
_generic_user_object
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
P
�0
�1
�2
�3
�4
�5"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�
jinputs
jmask
varargs
 
varkw
 
defaults�

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jinputs
jmask
varargs
 
varkw
 
defaults�

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
"
_generic_user_object
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper�
L__inference_ConcatenateLayer_layer_call_and_return_conditional_losses_233325�Z�W
P�M
K�H
"�
inputs_0���������	
"�
inputs_1���������
� ",�)
"�
tensor_0���������

� �
1__inference_ConcatenateLayer_layer_call_fn_233318Z�W
P�M
K�H
"�
inputs_0���������	
"�
inputs_1���������
� "!�
unknown���������
�
J__inference_FinalLayerNorm_layer_call_and_return_conditional_losses_233260kMN3�0
)�&
$�!
inputs���������P	
� "0�-
&�#
tensor_0���������P	
� �
/__inference_FinalLayerNorm_layer_call_fn_233218`MN3�0
)�&
$�!
inputs���������P	
� "%�"
unknown���������P	�
Z__inference_FullyConnectedLayerImprovement_layer_call_and_return_conditional_losses_233352cgh/�,
%�"
 �
inputs���������

� ",�)
"�
tensor_0���������

� �
?__inference_FullyConnectedLayerImprovement_layer_call_fn_233334Xgh/�,
%�"
 �
inputs���������

� "!�
unknown���������
�
H__inference_MaskingLayer_layer_call_and_return_conditional_losses_231889g3�0
)�&
$�!
inputs���������P	
� "0�-
&�#
tensor_0���������P	
� �
-__inference_MaskingLayer_layer_call_fn_231878\3�0
)�&
$�!
inputs���������P	
� "%�"
unknown���������P	�
B__inference_Output_layer_call_and_return_conditional_losses_233387X7�4
-�*
 �
inputs���������

 
p
� "�
�
tensor_0
� �
B__inference_Output_layer_call_and_return_conditional_losses_233392X7�4
-�*
 �
inputs���������

 
p 
� "�
�
tensor_0
� x
'__inference_Output_layer_call_fn_233377M7�4
-�*
 �
inputs���������

 
p
� "�
unknownx
'__inference_Output_layer_call_fn_233382M7�4
-�*
 �
inputs���������

 
p 
� "�
unknown�
Q__inference_PredictionImprovement_layer_call_and_return_conditional_losses_233372cop/�,
%�"
 �
inputs���������

� ",�)
"�
tensor_0���������
� �
6__inference_PredictionImprovement_layer_call_fn_233361Xop/�,
%�"
 �
inputs���������

� "!�
unknown����������
\__inference_ReduceStackDimensionViaSummation_layer_call_and_return_conditional_losses_233278k;�8
1�.
$�!
inputs���������P	

 
p
� ",�)
"�
tensor_0���������	
� �
\__inference_ReduceStackDimensionViaSummation_layer_call_and_return_conditional_losses_233286k;�8
1�.
$�!
inputs���������P	

 
p 
� ",�)
"�
tensor_0���������	
� �
A__inference_ReduceStackDimensionViaSummation_layer_call_fn_233265`;�8
1�.
$�!
inputs���������P	

 
p
� "!�
unknown���������	�
A__inference_ReduceStackDimensionViaSummation_layer_call_fn_233270`;�8
1�.
$�!
inputs���������P	

 
p 
� "!�
unknown���������	�
P__inference_StandardizeTimeLimit_layer_call_and_return_conditional_losses_233304g7�4
-�*
 �
inputs���������

 
p
� ",�)
"�
tensor_0���������
� �
P__inference_StandardizeTimeLimit_layer_call_and_return_conditional_losses_233312g7�4
-�*
 �
inputs���������

 
p 
� ",�)
"�
tensor_0���������
� �
5__inference_StandardizeTimeLimit_layer_call_fn_233291\7�4
-�*
 �
inputs���������

 
p
� "!�
unknown����������
5__inference_StandardizeTimeLimit_layer_call_fn_233296\7�4
-�*
 �
inputs���������

 
p 
� "!�
unknown����������
!__inference__wrapped_model_229753�]�wxyz{|}~��������������������������������������MNghops�p
i�f
d�a
5�2
StackLevelInputFeatures���������P	
(�%
TimeLimitInput���������
� " �

Output�
output�
C__inference_model_1_layer_call_and_return_conditional_losses_230556�]�wxyz{|}~��������������������������������������MNghop{�x
q�n
d�a
5�2
StackLevelInputFeatures���������P	
(�%
TimeLimitInput���������
p

 
� "�
�
tensor_0
� �
C__inference_model_1_layer_call_and_return_conditional_losses_231228�]�wxyz{|}~��������������������������������������MNghop{�x
q�n
d�a
5�2
StackLevelInputFeatures���������P	
(�%
TimeLimitInput���������
p 

 
� "�
�
tensor_0
� �
(__inference_model_1_layer_call_fn_231342�]�wxyz{|}~��������������������������������������MNghop{�x
q�n
d�a
5�2
StackLevelInputFeatures���������P	
(�%
TimeLimitInput���������
p

 
� "�
unknown�
(__inference_model_1_layer_call_fn_231456�]�wxyz{|}~��������������������������������������MNghop{�x
q�n
d�a
5�2
StackLevelInputFeatures���������P	
(�%
TimeLimitInput���������
p 

 
� "�
unknown�
$__inference_signature_wrapper_231873�]�wxyz{|}~��������������������������������������MNghop���
� 
���
P
StackLevelInputFeatures5�2
stacklevelinputfeatures���������P	
:
TimeLimitInput(�%
timelimitinput���������" �

Output�
output�
Q__inference_transformer_encoder_3_layer_call_and_return_conditional_losses_232155��wxyz{|}~������C�@
)�&
$�!
inputs���������P	
�

trainingp"e�b
[�X
(�%

tensor_0_0���������P	
,�)

tensor_0_1���������PP
� �
Q__inference_transformer_encoder_3_layer_call_and_return_conditional_losses_232329��wxyz{|}~������C�@
)�&
$�!
inputs���������P	
�

trainingp "e�b
[�X
(�%

tensor_0_0���������P	
,�)

tensor_0_1���������PP
� �
6__inference_transformer_encoder_3_layer_call_fn_231928��wxyz{|}~������C�@
)�&
$�!
inputs���������P	
�

trainingp"W�T
&�#
tensor_0���������P	
*�'
tensor_1���������PP�
6__inference_transformer_encoder_3_layer_call_fn_231967��wxyz{|}~������C�@
)�&
$�!
inputs���������P	
�

trainingp "W�T
&�#
tensor_0���������P	
*�'
tensor_1���������PP�
Q__inference_transformer_encoder_4_layer_call_and_return_conditional_losses_232595� ����������������C�@
)�&
$�!
inputs���������P	
�

trainingp"e�b
[�X
(�%

tensor_0_0���������P	
,�)

tensor_0_1���������PP
� �
Q__inference_transformer_encoder_4_layer_call_and_return_conditional_losses_232769� ����������������C�@
)�&
$�!
inputs���������P	
�

trainingp "e�b
[�X
(�%

tensor_0_0���������P	
,�)

tensor_0_1���������PP
� �
6__inference_transformer_encoder_4_layer_call_fn_232368� ����������������C�@
)�&
$�!
inputs���������P	
�

trainingp"W�T
&�#
tensor_0���������P	
*�'
tensor_1���������PP�
6__inference_transformer_encoder_4_layer_call_fn_232407� ����������������C�@
)�&
$�!
inputs���������P	
�

trainingp "W�T
&�#
tensor_0���������P	
*�'
tensor_1���������PP�
Q__inference_transformer_encoder_5_layer_call_and_return_conditional_losses_233035� ����������������C�@
)�&
$�!
inputs���������P	
�

trainingp"e�b
[�X
(�%

tensor_0_0���������P	
,�)

tensor_0_1���������PP
� �
Q__inference_transformer_encoder_5_layer_call_and_return_conditional_losses_233209� ����������������C�@
)�&
$�!
inputs���������P	
�

trainingp "e�b
[�X
(�%

tensor_0_0���������P	
,�)

tensor_0_1���������PP
� �
6__inference_transformer_encoder_5_layer_call_fn_232808� ����������������C�@
)�&
$�!
inputs���������P	
�

trainingp"W�T
&�#
tensor_0���������P	
*�'
tensor_1���������PP�
6__inference_transformer_encoder_5_layer_call_fn_232847� ����������������C�@
)�&
$�!
inputs���������P	
�

trainingp "W�T
&�#
tensor_0���������P	
*�'
tensor_1���������PP