��:
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
 �"serve*2.14.02v2.14.0-rc1-21-g4dacf3f368e8��4
�
8transformer_encoder_17/Encoder-FeedForwardLayer_2_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*I
shared_name:8transformer_encoder_17/Encoder-FeedForwardLayer_2_3/bias
�
Ltransformer_encoder_17/Encoder-FeedForwardLayer_2_3/bias/Read/ReadVariableOpReadVariableOp8transformer_encoder_17/Encoder-FeedForwardLayer_2_3/bias*
_output_shapes
:	*
dtype0
�
:transformer_encoder_17/Encoder-FeedForwardLayer_2_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:		*K
shared_name<:transformer_encoder_17/Encoder-FeedForwardLayer_2_3/kernel
�
Ntransformer_encoder_17/Encoder-FeedForwardLayer_2_3/kernel/Read/ReadVariableOpReadVariableOp:transformer_encoder_17/Encoder-FeedForwardLayer_2_3/kernel*
_output_shapes

:		*
dtype0
�
8transformer_encoder_17/Encoder-FeedForwardLayer_1_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*I
shared_name:8transformer_encoder_17/Encoder-FeedForwardLayer_1_3/bias
�
Ltransformer_encoder_17/Encoder-FeedForwardLayer_1_3/bias/Read/ReadVariableOpReadVariableOp8transformer_encoder_17/Encoder-FeedForwardLayer_1_3/bias*
_output_shapes
:	*
dtype0
�
:transformer_encoder_17/Encoder-FeedForwardLayer_1_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:		*K
shared_name<:transformer_encoder_17/Encoder-FeedForwardLayer_1_3/kernel
�
Ntransformer_encoder_17/Encoder-FeedForwardLayer_1_3/kernel/Read/ReadVariableOpReadVariableOp:transformer_encoder_17/Encoder-FeedForwardLayer_1_3/kernel*
_output_shapes

:		*
dtype0
�
<transformer_encoder_17/Encoder-2nd-NormalizationLayer-3/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*M
shared_name><transformer_encoder_17/Encoder-2nd-NormalizationLayer-3/beta
�
Ptransformer_encoder_17/Encoder-2nd-NormalizationLayer-3/beta/Read/ReadVariableOpReadVariableOp<transformer_encoder_17/Encoder-2nd-NormalizationLayer-3/beta*
_output_shapes
:	*
dtype0
�
=transformer_encoder_17/Encoder-2nd-NormalizationLayer-3/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*N
shared_name?=transformer_encoder_17/Encoder-2nd-NormalizationLayer-3/gamma
�
Qtransformer_encoder_17/Encoder-2nd-NormalizationLayer-3/gamma/Read/ReadVariableOpReadVariableOp=transformer_encoder_17/Encoder-2nd-NormalizationLayer-3/gamma*
_output_shapes
:	*
dtype0
�
<transformer_encoder_17/Encoder-1st-NormalizationLayer-3/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*M
shared_name><transformer_encoder_17/Encoder-1st-NormalizationLayer-3/beta
�
Ptransformer_encoder_17/Encoder-1st-NormalizationLayer-3/beta/Read/ReadVariableOpReadVariableOp<transformer_encoder_17/Encoder-1st-NormalizationLayer-3/beta*
_output_shapes
:	*
dtype0
�
=transformer_encoder_17/Encoder-1st-NormalizationLayer-3/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*N
shared_name?=transformer_encoder_17/Encoder-1st-NormalizationLayer-3/gamma
�
Qtransformer_encoder_17/Encoder-1st-NormalizationLayer-3/gamma/Read/ReadVariableOpReadVariableOp=transformer_encoder_17/Encoder-1st-NormalizationLayer-3/gamma*
_output_shapes
:	*
dtype0
�
Itransformer_encoder_17/Encoder-SelfAttentionLayer-3/attention_output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*Z
shared_nameKItransformer_encoder_17/Encoder-SelfAttentionLayer-3/attention_output/bias
�
]transformer_encoder_17/Encoder-SelfAttentionLayer-3/attention_output/bias/Read/ReadVariableOpReadVariableOpItransformer_encoder_17/Encoder-SelfAttentionLayer-3/attention_output/bias*
_output_shapes
:	*
dtype0
�
Ktransformer_encoder_17/Encoder-SelfAttentionLayer-3/attention_output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*\
shared_nameMKtransformer_encoder_17/Encoder-SelfAttentionLayer-3/attention_output/kernel
�
_transformer_encoder_17/Encoder-SelfAttentionLayer-3/attention_output/kernel/Read/ReadVariableOpReadVariableOpKtransformer_encoder_17/Encoder-SelfAttentionLayer-3/attention_output/kernel*"
_output_shapes
:	*
dtype0
�
>transformer_encoder_17/Encoder-SelfAttentionLayer-3/value/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*O
shared_name@>transformer_encoder_17/Encoder-SelfAttentionLayer-3/value/bias
�
Rtransformer_encoder_17/Encoder-SelfAttentionLayer-3/value/bias/Read/ReadVariableOpReadVariableOp>transformer_encoder_17/Encoder-SelfAttentionLayer-3/value/bias*
_output_shapes

:*
dtype0
�
@transformer_encoder_17/Encoder-SelfAttentionLayer-3/value/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*Q
shared_nameB@transformer_encoder_17/Encoder-SelfAttentionLayer-3/value/kernel
�
Ttransformer_encoder_17/Encoder-SelfAttentionLayer-3/value/kernel/Read/ReadVariableOpReadVariableOp@transformer_encoder_17/Encoder-SelfAttentionLayer-3/value/kernel*"
_output_shapes
:	*
dtype0
�
<transformer_encoder_17/Encoder-SelfAttentionLayer-3/key/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*M
shared_name><transformer_encoder_17/Encoder-SelfAttentionLayer-3/key/bias
�
Ptransformer_encoder_17/Encoder-SelfAttentionLayer-3/key/bias/Read/ReadVariableOpReadVariableOp<transformer_encoder_17/Encoder-SelfAttentionLayer-3/key/bias*
_output_shapes

:*
dtype0
�
>transformer_encoder_17/Encoder-SelfAttentionLayer-3/key/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*O
shared_name@>transformer_encoder_17/Encoder-SelfAttentionLayer-3/key/kernel
�
Rtransformer_encoder_17/Encoder-SelfAttentionLayer-3/key/kernel/Read/ReadVariableOpReadVariableOp>transformer_encoder_17/Encoder-SelfAttentionLayer-3/key/kernel*"
_output_shapes
:	*
dtype0
�
>transformer_encoder_17/Encoder-SelfAttentionLayer-3/query/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*O
shared_name@>transformer_encoder_17/Encoder-SelfAttentionLayer-3/query/bias
�
Rtransformer_encoder_17/Encoder-SelfAttentionLayer-3/query/bias/Read/ReadVariableOpReadVariableOp>transformer_encoder_17/Encoder-SelfAttentionLayer-3/query/bias*
_output_shapes

:*
dtype0
�
@transformer_encoder_17/Encoder-SelfAttentionLayer-3/query/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*Q
shared_nameB@transformer_encoder_17/Encoder-SelfAttentionLayer-3/query/kernel
�
Ttransformer_encoder_17/Encoder-SelfAttentionLayer-3/query/kernel/Read/ReadVariableOpReadVariableOp@transformer_encoder_17/Encoder-SelfAttentionLayer-3/query/kernel*"
_output_shapes
:	*
dtype0
�
8transformer_encoder_16/Encoder-FeedForwardLayer_2_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*I
shared_name:8transformer_encoder_16/Encoder-FeedForwardLayer_2_2/bias
�
Ltransformer_encoder_16/Encoder-FeedForwardLayer_2_2/bias/Read/ReadVariableOpReadVariableOp8transformer_encoder_16/Encoder-FeedForwardLayer_2_2/bias*
_output_shapes
:	*
dtype0
�
:transformer_encoder_16/Encoder-FeedForwardLayer_2_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:		*K
shared_name<:transformer_encoder_16/Encoder-FeedForwardLayer_2_2/kernel
�
Ntransformer_encoder_16/Encoder-FeedForwardLayer_2_2/kernel/Read/ReadVariableOpReadVariableOp:transformer_encoder_16/Encoder-FeedForwardLayer_2_2/kernel*
_output_shapes

:		*
dtype0
�
8transformer_encoder_16/Encoder-FeedForwardLayer_1_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*I
shared_name:8transformer_encoder_16/Encoder-FeedForwardLayer_1_2/bias
�
Ltransformer_encoder_16/Encoder-FeedForwardLayer_1_2/bias/Read/ReadVariableOpReadVariableOp8transformer_encoder_16/Encoder-FeedForwardLayer_1_2/bias*
_output_shapes
:	*
dtype0
�
:transformer_encoder_16/Encoder-FeedForwardLayer_1_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:		*K
shared_name<:transformer_encoder_16/Encoder-FeedForwardLayer_1_2/kernel
�
Ntransformer_encoder_16/Encoder-FeedForwardLayer_1_2/kernel/Read/ReadVariableOpReadVariableOp:transformer_encoder_16/Encoder-FeedForwardLayer_1_2/kernel*
_output_shapes

:		*
dtype0
�
<transformer_encoder_16/Encoder-2nd-NormalizationLayer-2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*M
shared_name><transformer_encoder_16/Encoder-2nd-NormalizationLayer-2/beta
�
Ptransformer_encoder_16/Encoder-2nd-NormalizationLayer-2/beta/Read/ReadVariableOpReadVariableOp<transformer_encoder_16/Encoder-2nd-NormalizationLayer-2/beta*
_output_shapes
:	*
dtype0
�
=transformer_encoder_16/Encoder-2nd-NormalizationLayer-2/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*N
shared_name?=transformer_encoder_16/Encoder-2nd-NormalizationLayer-2/gamma
�
Qtransformer_encoder_16/Encoder-2nd-NormalizationLayer-2/gamma/Read/ReadVariableOpReadVariableOp=transformer_encoder_16/Encoder-2nd-NormalizationLayer-2/gamma*
_output_shapes
:	*
dtype0
�
<transformer_encoder_16/Encoder-1st-NormalizationLayer-2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*M
shared_name><transformer_encoder_16/Encoder-1st-NormalizationLayer-2/beta
�
Ptransformer_encoder_16/Encoder-1st-NormalizationLayer-2/beta/Read/ReadVariableOpReadVariableOp<transformer_encoder_16/Encoder-1st-NormalizationLayer-2/beta*
_output_shapes
:	*
dtype0
�
=transformer_encoder_16/Encoder-1st-NormalizationLayer-2/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*N
shared_name?=transformer_encoder_16/Encoder-1st-NormalizationLayer-2/gamma
�
Qtransformer_encoder_16/Encoder-1st-NormalizationLayer-2/gamma/Read/ReadVariableOpReadVariableOp=transformer_encoder_16/Encoder-1st-NormalizationLayer-2/gamma*
_output_shapes
:	*
dtype0
�
Itransformer_encoder_16/Encoder-SelfAttentionLayer-2/attention_output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*Z
shared_nameKItransformer_encoder_16/Encoder-SelfAttentionLayer-2/attention_output/bias
�
]transformer_encoder_16/Encoder-SelfAttentionLayer-2/attention_output/bias/Read/ReadVariableOpReadVariableOpItransformer_encoder_16/Encoder-SelfAttentionLayer-2/attention_output/bias*
_output_shapes
:	*
dtype0
�
Ktransformer_encoder_16/Encoder-SelfAttentionLayer-2/attention_output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*\
shared_nameMKtransformer_encoder_16/Encoder-SelfAttentionLayer-2/attention_output/kernel
�
_transformer_encoder_16/Encoder-SelfAttentionLayer-2/attention_output/kernel/Read/ReadVariableOpReadVariableOpKtransformer_encoder_16/Encoder-SelfAttentionLayer-2/attention_output/kernel*"
_output_shapes
:	*
dtype0
�
>transformer_encoder_16/Encoder-SelfAttentionLayer-2/value/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*O
shared_name@>transformer_encoder_16/Encoder-SelfAttentionLayer-2/value/bias
�
Rtransformer_encoder_16/Encoder-SelfAttentionLayer-2/value/bias/Read/ReadVariableOpReadVariableOp>transformer_encoder_16/Encoder-SelfAttentionLayer-2/value/bias*
_output_shapes

:*
dtype0
�
@transformer_encoder_16/Encoder-SelfAttentionLayer-2/value/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*Q
shared_nameB@transformer_encoder_16/Encoder-SelfAttentionLayer-2/value/kernel
�
Ttransformer_encoder_16/Encoder-SelfAttentionLayer-2/value/kernel/Read/ReadVariableOpReadVariableOp@transformer_encoder_16/Encoder-SelfAttentionLayer-2/value/kernel*"
_output_shapes
:	*
dtype0
�
<transformer_encoder_16/Encoder-SelfAttentionLayer-2/key/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*M
shared_name><transformer_encoder_16/Encoder-SelfAttentionLayer-2/key/bias
�
Ptransformer_encoder_16/Encoder-SelfAttentionLayer-2/key/bias/Read/ReadVariableOpReadVariableOp<transformer_encoder_16/Encoder-SelfAttentionLayer-2/key/bias*
_output_shapes

:*
dtype0
�
>transformer_encoder_16/Encoder-SelfAttentionLayer-2/key/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*O
shared_name@>transformer_encoder_16/Encoder-SelfAttentionLayer-2/key/kernel
�
Rtransformer_encoder_16/Encoder-SelfAttentionLayer-2/key/kernel/Read/ReadVariableOpReadVariableOp>transformer_encoder_16/Encoder-SelfAttentionLayer-2/key/kernel*"
_output_shapes
:	*
dtype0
�
>transformer_encoder_16/Encoder-SelfAttentionLayer-2/query/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*O
shared_name@>transformer_encoder_16/Encoder-SelfAttentionLayer-2/query/bias
�
Rtransformer_encoder_16/Encoder-SelfAttentionLayer-2/query/bias/Read/ReadVariableOpReadVariableOp>transformer_encoder_16/Encoder-SelfAttentionLayer-2/query/bias*
_output_shapes

:*
dtype0
�
@transformer_encoder_16/Encoder-SelfAttentionLayer-2/query/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*Q
shared_nameB@transformer_encoder_16/Encoder-SelfAttentionLayer-2/query/kernel
�
Ttransformer_encoder_16/Encoder-SelfAttentionLayer-2/query/kernel/Read/ReadVariableOpReadVariableOp@transformer_encoder_16/Encoder-SelfAttentionLayer-2/query/kernel*"
_output_shapes
:	*
dtype0
�
8transformer_encoder_15/Encoder-FeedForwardLayer_2_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*I
shared_name:8transformer_encoder_15/Encoder-FeedForwardLayer_2_1/bias
�
Ltransformer_encoder_15/Encoder-FeedForwardLayer_2_1/bias/Read/ReadVariableOpReadVariableOp8transformer_encoder_15/Encoder-FeedForwardLayer_2_1/bias*
_output_shapes
:	*
dtype0
�
:transformer_encoder_15/Encoder-FeedForwardLayer_2_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:		*K
shared_name<:transformer_encoder_15/Encoder-FeedForwardLayer_2_1/kernel
�
Ntransformer_encoder_15/Encoder-FeedForwardLayer_2_1/kernel/Read/ReadVariableOpReadVariableOp:transformer_encoder_15/Encoder-FeedForwardLayer_2_1/kernel*
_output_shapes

:		*
dtype0
�
8transformer_encoder_15/Encoder-FeedForwardLayer_1_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*I
shared_name:8transformer_encoder_15/Encoder-FeedForwardLayer_1_1/bias
�
Ltransformer_encoder_15/Encoder-FeedForwardLayer_1_1/bias/Read/ReadVariableOpReadVariableOp8transformer_encoder_15/Encoder-FeedForwardLayer_1_1/bias*
_output_shapes
:	*
dtype0
�
:transformer_encoder_15/Encoder-FeedForwardLayer_1_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:		*K
shared_name<:transformer_encoder_15/Encoder-FeedForwardLayer_1_1/kernel
�
Ntransformer_encoder_15/Encoder-FeedForwardLayer_1_1/kernel/Read/ReadVariableOpReadVariableOp:transformer_encoder_15/Encoder-FeedForwardLayer_1_1/kernel*
_output_shapes

:		*
dtype0
�
<transformer_encoder_15/Encoder-2nd-NormalizationLayer-1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*M
shared_name><transformer_encoder_15/Encoder-2nd-NormalizationLayer-1/beta
�
Ptransformer_encoder_15/Encoder-2nd-NormalizationLayer-1/beta/Read/ReadVariableOpReadVariableOp<transformer_encoder_15/Encoder-2nd-NormalizationLayer-1/beta*
_output_shapes
:	*
dtype0
�
=transformer_encoder_15/Encoder-2nd-NormalizationLayer-1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*N
shared_name?=transformer_encoder_15/Encoder-2nd-NormalizationLayer-1/gamma
�
Qtransformer_encoder_15/Encoder-2nd-NormalizationLayer-1/gamma/Read/ReadVariableOpReadVariableOp=transformer_encoder_15/Encoder-2nd-NormalizationLayer-1/gamma*
_output_shapes
:	*
dtype0
�
<transformer_encoder_15/Encoder-1st-NormalizationLayer-1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*M
shared_name><transformer_encoder_15/Encoder-1st-NormalizationLayer-1/beta
�
Ptransformer_encoder_15/Encoder-1st-NormalizationLayer-1/beta/Read/ReadVariableOpReadVariableOp<transformer_encoder_15/Encoder-1st-NormalizationLayer-1/beta*
_output_shapes
:	*
dtype0
�
=transformer_encoder_15/Encoder-1st-NormalizationLayer-1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*N
shared_name?=transformer_encoder_15/Encoder-1st-NormalizationLayer-1/gamma
�
Qtransformer_encoder_15/Encoder-1st-NormalizationLayer-1/gamma/Read/ReadVariableOpReadVariableOp=transformer_encoder_15/Encoder-1st-NormalizationLayer-1/gamma*
_output_shapes
:	*
dtype0
�
Itransformer_encoder_15/Encoder-SelfAttentionLayer-1/attention_output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*Z
shared_nameKItransformer_encoder_15/Encoder-SelfAttentionLayer-1/attention_output/bias
�
]transformer_encoder_15/Encoder-SelfAttentionLayer-1/attention_output/bias/Read/ReadVariableOpReadVariableOpItransformer_encoder_15/Encoder-SelfAttentionLayer-1/attention_output/bias*
_output_shapes
:	*
dtype0
�
Ktransformer_encoder_15/Encoder-SelfAttentionLayer-1/attention_output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*\
shared_nameMKtransformer_encoder_15/Encoder-SelfAttentionLayer-1/attention_output/kernel
�
_transformer_encoder_15/Encoder-SelfAttentionLayer-1/attention_output/kernel/Read/ReadVariableOpReadVariableOpKtransformer_encoder_15/Encoder-SelfAttentionLayer-1/attention_output/kernel*"
_output_shapes
:	*
dtype0
�
>transformer_encoder_15/Encoder-SelfAttentionLayer-1/value/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*O
shared_name@>transformer_encoder_15/Encoder-SelfAttentionLayer-1/value/bias
�
Rtransformer_encoder_15/Encoder-SelfAttentionLayer-1/value/bias/Read/ReadVariableOpReadVariableOp>transformer_encoder_15/Encoder-SelfAttentionLayer-1/value/bias*
_output_shapes

:*
dtype0
�
@transformer_encoder_15/Encoder-SelfAttentionLayer-1/value/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*Q
shared_nameB@transformer_encoder_15/Encoder-SelfAttentionLayer-1/value/kernel
�
Ttransformer_encoder_15/Encoder-SelfAttentionLayer-1/value/kernel/Read/ReadVariableOpReadVariableOp@transformer_encoder_15/Encoder-SelfAttentionLayer-1/value/kernel*"
_output_shapes
:	*
dtype0
�
<transformer_encoder_15/Encoder-SelfAttentionLayer-1/key/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*M
shared_name><transformer_encoder_15/Encoder-SelfAttentionLayer-1/key/bias
�
Ptransformer_encoder_15/Encoder-SelfAttentionLayer-1/key/bias/Read/ReadVariableOpReadVariableOp<transformer_encoder_15/Encoder-SelfAttentionLayer-1/key/bias*
_output_shapes

:*
dtype0
�
>transformer_encoder_15/Encoder-SelfAttentionLayer-1/key/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*O
shared_name@>transformer_encoder_15/Encoder-SelfAttentionLayer-1/key/kernel
�
Rtransformer_encoder_15/Encoder-SelfAttentionLayer-1/key/kernel/Read/ReadVariableOpReadVariableOp>transformer_encoder_15/Encoder-SelfAttentionLayer-1/key/kernel*"
_output_shapes
:	*
dtype0
�
>transformer_encoder_15/Encoder-SelfAttentionLayer-1/query/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*O
shared_name@>transformer_encoder_15/Encoder-SelfAttentionLayer-1/query/bias
�
Rtransformer_encoder_15/Encoder-SelfAttentionLayer-1/query/bias/Read/ReadVariableOpReadVariableOp>transformer_encoder_15/Encoder-SelfAttentionLayer-1/query/bias*
_output_shapes

:*
dtype0
�
@transformer_encoder_15/Encoder-SelfAttentionLayer-1/query/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*Q
shared_nameB@transformer_encoder_15/Encoder-SelfAttentionLayer-1/query/kernel
�
Ttransformer_encoder_15/Encoder-SelfAttentionLayer-1/query/kernel/Read/ReadVariableOpReadVariableOp@transformer_encoder_15/Encoder-SelfAttentionLayer-1/query/kernel*"
_output_shapes
:	*
dtype0
�
PredictionSolved/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_namePredictionSolved/bias
{
)PredictionSolved/bias/Read/ReadVariableOpReadVariableOpPredictionSolved/bias*
_output_shapes
:*
dtype0
�
PredictionSolved/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*(
shared_namePredictionSolved/kernel
�
+PredictionSolved/kernel/Read/ReadVariableOpReadVariableOpPredictionSolved/kernel*
_output_shapes

:
*
dtype0
�
PredictionAreaRatio/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_namePredictionAreaRatio/bias
�
,PredictionAreaRatio/bias/Read/ReadVariableOpReadVariableOpPredictionAreaRatio/bias*
_output_shapes
:*
dtype0
�
PredictionAreaRatio/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*+
shared_namePredictionAreaRatio/kernel
�
.PredictionAreaRatio/kernel/Read/ReadVariableOpReadVariableOpPredictionAreaRatio/kernel*
_output_shapes

:
*
dtype0
�
FullyConnectedLayerSolved/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*/
shared_name FullyConnectedLayerSolved/bias
�
2FullyConnectedLayerSolved/bias/Read/ReadVariableOpReadVariableOpFullyConnectedLayerSolved/bias*
_output_shapes
:
*
dtype0
�
 FullyConnectedLayerSolved/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:

*1
shared_name" FullyConnectedLayerSolved/kernel
�
4FullyConnectedLayerSolved/kernel/Read/ReadVariableOpReadVariableOp FullyConnectedLayerSolved/kernel*
_output_shapes

:

*
dtype0
�
!FullyConnectedLayerAreaRatio/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*2
shared_name#!FullyConnectedLayerAreaRatio/bias
�
5FullyConnectedLayerAreaRatio/bias/Read/ReadVariableOpReadVariableOp!FullyConnectedLayerAreaRatio/bias*
_output_shapes
:
*
dtype0
�
#FullyConnectedLayerAreaRatio/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:

*4
shared_name%#FullyConnectedLayerAreaRatio/kernel
�
7FullyConnectedLayerAreaRatio/kernel/Read/ReadVariableOpReadVariableOp#FullyConnectedLayerAreaRatio/kernel*
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
StatefulPartitionedCallStatefulPartitionedCall'serving_default_StackLevelInputFeaturesserving_default_TimeLimitInput=transformer_encoder_15/Encoder-1st-NormalizationLayer-1/gamma<transformer_encoder_15/Encoder-1st-NormalizationLayer-1/beta@transformer_encoder_15/Encoder-SelfAttentionLayer-1/query/kernel>transformer_encoder_15/Encoder-SelfAttentionLayer-1/query/bias>transformer_encoder_15/Encoder-SelfAttentionLayer-1/key/kernel<transformer_encoder_15/Encoder-SelfAttentionLayer-1/key/bias@transformer_encoder_15/Encoder-SelfAttentionLayer-1/value/kernel>transformer_encoder_15/Encoder-SelfAttentionLayer-1/value/biasKtransformer_encoder_15/Encoder-SelfAttentionLayer-1/attention_output/kernelItransformer_encoder_15/Encoder-SelfAttentionLayer-1/attention_output/bias=transformer_encoder_15/Encoder-2nd-NormalizationLayer-1/gamma<transformer_encoder_15/Encoder-2nd-NormalizationLayer-1/beta:transformer_encoder_15/Encoder-FeedForwardLayer_1_1/kernel8transformer_encoder_15/Encoder-FeedForwardLayer_1_1/bias:transformer_encoder_15/Encoder-FeedForwardLayer_2_1/kernel8transformer_encoder_15/Encoder-FeedForwardLayer_2_1/bias=transformer_encoder_16/Encoder-1st-NormalizationLayer-2/gamma<transformer_encoder_16/Encoder-1st-NormalizationLayer-2/beta@transformer_encoder_16/Encoder-SelfAttentionLayer-2/query/kernel>transformer_encoder_16/Encoder-SelfAttentionLayer-2/query/bias>transformer_encoder_16/Encoder-SelfAttentionLayer-2/key/kernel<transformer_encoder_16/Encoder-SelfAttentionLayer-2/key/bias@transformer_encoder_16/Encoder-SelfAttentionLayer-2/value/kernel>transformer_encoder_16/Encoder-SelfAttentionLayer-2/value/biasKtransformer_encoder_16/Encoder-SelfAttentionLayer-2/attention_output/kernelItransformer_encoder_16/Encoder-SelfAttentionLayer-2/attention_output/bias=transformer_encoder_16/Encoder-2nd-NormalizationLayer-2/gamma<transformer_encoder_16/Encoder-2nd-NormalizationLayer-2/beta:transformer_encoder_16/Encoder-FeedForwardLayer_1_2/kernel8transformer_encoder_16/Encoder-FeedForwardLayer_1_2/bias:transformer_encoder_16/Encoder-FeedForwardLayer_2_2/kernel8transformer_encoder_16/Encoder-FeedForwardLayer_2_2/bias=transformer_encoder_17/Encoder-1st-NormalizationLayer-3/gamma<transformer_encoder_17/Encoder-1st-NormalizationLayer-3/beta@transformer_encoder_17/Encoder-SelfAttentionLayer-3/query/kernel>transformer_encoder_17/Encoder-SelfAttentionLayer-3/query/bias>transformer_encoder_17/Encoder-SelfAttentionLayer-3/key/kernel<transformer_encoder_17/Encoder-SelfAttentionLayer-3/key/bias@transformer_encoder_17/Encoder-SelfAttentionLayer-3/value/kernel>transformer_encoder_17/Encoder-SelfAttentionLayer-3/value/biasKtransformer_encoder_17/Encoder-SelfAttentionLayer-3/attention_output/kernelItransformer_encoder_17/Encoder-SelfAttentionLayer-3/attention_output/bias=transformer_encoder_17/Encoder-2nd-NormalizationLayer-3/gamma<transformer_encoder_17/Encoder-2nd-NormalizationLayer-3/beta:transformer_encoder_17/Encoder-FeedForwardLayer_1_3/kernel8transformer_encoder_17/Encoder-FeedForwardLayer_1_3/bias:transformer_encoder_17/Encoder-FeedForwardLayer_2_3/kernel8transformer_encoder_17/Encoder-FeedForwardLayer_2_3/biasFinalLayerNorm/gammaFinalLayerNorm/beta FullyConnectedLayerSolved/kernelFullyConnectedLayerSolved/bias#FullyConnectedLayerAreaRatio/kernel!FullyConnectedLayerAreaRatio/biasPredictionSolved/kernelPredictionSolved/biasPredictionAreaRatio/kernelPredictionAreaRatio/bias*G
Tin@
>2<*
Tout
2*
_collective_manager_ids
 *
_output_shapes

::*\
_read_only_resource_inputs>
<:	
 !"#$%&'()*+,-./0123456789:;*0
config_proto 

CPU

GPU2*0J 8� *.
f)R'
%__inference_signature_wrapper_1431389

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
layer_with_weights-6
layer-12
layer_with_weights-7
layer-13
layer-14
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures*
* 
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses* 
�
	variables
trainable_variables
 regularization_losses
!	keras_api
"__call__
*#&call_and_return_all_conditional_losses
$self_attention_layer
%add1
&add2
'
layernorm1
(
layernorm2
)feed_forward_layer_1
*feed_forward_layer_2
+dropout_layer*
�
,	variables
-trainable_variables
.regularization_losses
/	keras_api
0__call__
*1&call_and_return_all_conditional_losses
2self_attention_layer
3add1
4add2
5
layernorm1
6
layernorm2
7feed_forward_layer_1
8feed_forward_layer_2
9dropout_layer*
�
:	variables
;trainable_variables
<regularization_losses
=	keras_api
>__call__
*?&call_and_return_all_conditional_losses
@self_attention_layer
Aadd1
Badd2
C
layernorm1
D
layernorm2
Efeed_forward_layer_1
Ffeed_forward_layer_2
Gdropout_layer*
�
H	variables
Itrainable_variables
Jregularization_losses
K	keras_api
L__call__
*M&call_and_return_all_conditional_losses
Naxis
	Ogamma
Pbeta*
* 
�
Q	variables
Rtrainable_variables
Sregularization_losses
T	keras_api
U__call__
*V&call_and_return_all_conditional_losses* 
�
W	variables
Xtrainable_variables
Yregularization_losses
Z	keras_api
[__call__
*\&call_and_return_all_conditional_losses* 
�
]	variables
^trainable_variables
_regularization_losses
`	keras_api
a__call__
*b&call_and_return_all_conditional_losses* 
�
c	variables
dtrainable_variables
eregularization_losses
f	keras_api
g__call__
*h&call_and_return_all_conditional_losses

ikernel
jbias*
�
k	variables
ltrainable_variables
mregularization_losses
n	keras_api
o__call__
*p&call_and_return_all_conditional_losses

qkernel
rbias*
�
s	variables
ttrainable_variables
uregularization_losses
v	keras_api
w__call__
*x&call_and_return_all_conditional_losses

ykernel
zbias*
�
{	variables
|trainable_variables
}regularization_losses
~	keras_api
__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�
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
O48
P49
i50
j51
q52
r53
y54
z55
�56
�57*
�
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
O48
P49
i50
j51
q52
r53
y54
z55
�56
�57*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

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
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 
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
	variables
trainable_variables
 regularization_losses
"__call__
*#&call_and_return_all_conditional_losses
&#"call_and_return_conditional_losses*

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

�gamma
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
,	variables
-trainable_variables
.regularization_losses
0__call__
*1&call_and_return_all_conditional_losses
&1"call_and_return_conditional_losses*

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
:	variables
;trainable_variables
<regularization_losses
>__call__
*?&call_and_return_all_conditional_losses
&?"call_and_return_conditional_losses*

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
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�_random_generator* 

O0
P1*

O0
P1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
H	variables
Itrainable_variables
Jregularization_losses
L__call__
*M&call_and_return_all_conditional_losses
&M"call_and_return_conditional_losses*
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
Q	variables
Rtrainable_variables
Sregularization_losses
U__call__
*V&call_and_return_all_conditional_losses
&V"call_and_return_conditional_losses* 

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
W	variables
Xtrainable_variables
Yregularization_losses
[__call__
*\&call_and_return_all_conditional_losses
&\"call_and_return_conditional_losses* 

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
]	variables
^trainable_variables
_regularization_losses
a__call__
*b&call_and_return_all_conditional_losses
&b"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

i0
j1*

i0
j1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
c	variables
dtrainable_variables
eregularization_losses
g__call__
*h&call_and_return_all_conditional_losses
&h"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
sm
VARIABLE_VALUE#FullyConnectedLayerAreaRatio/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
oi
VARIABLE_VALUE!FullyConnectedLayerAreaRatio/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*

q0
r1*

q0
r1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
k	variables
ltrainable_variables
mregularization_losses
o__call__
*p&call_and_return_all_conditional_losses
&p"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
pj
VARIABLE_VALUE FullyConnectedLayerSolved/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE*
lf
VARIABLE_VALUEFullyConnectedLayerSolved/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE*

y0
z1*

y0
z1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
s	variables
ttrainable_variables
uregularization_losses
w__call__
*x&call_and_return_all_conditional_losses
&x"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
jd
VARIABLE_VALUEPredictionAreaRatio/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUEPredictionAreaRatio/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE*
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
{	variables
|trainable_variables
}regularization_losses
__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
ga
VARIABLE_VALUEPredictionSolved/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEPredictionSolved/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE*
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

�trace_0
�trace_1* 

�trace_0
�trace_1* 
�z
VARIABLE_VALUE@transformer_encoder_15/Encoder-SelfAttentionLayer-1/query/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE>transformer_encoder_15/Encoder-SelfAttentionLayer-1/query/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE>transformer_encoder_15/Encoder-SelfAttentionLayer-1/key/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUE<transformer_encoder_15/Encoder-SelfAttentionLayer-1/key/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
�z
VARIABLE_VALUE@transformer_encoder_15/Encoder-SelfAttentionLayer-1/value/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE>transformer_encoder_15/Encoder-SelfAttentionLayer-1/value/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEKtransformer_encoder_15/Encoder-SelfAttentionLayer-1/attention_output/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEItransformer_encoder_15/Encoder-SelfAttentionLayer-1/attention_output/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUE=transformer_encoder_15/Encoder-1st-NormalizationLayer-1/gamma&variables/8/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUE<transformer_encoder_15/Encoder-1st-NormalizationLayer-1/beta&variables/9/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE=transformer_encoder_15/Encoder-2nd-NormalizationLayer-1/gamma'variables/10/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUE<transformer_encoder_15/Encoder-2nd-NormalizationLayer-1/beta'variables/11/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUE:transformer_encoder_15/Encoder-FeedForwardLayer_1_1/kernel'variables/12/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUE8transformer_encoder_15/Encoder-FeedForwardLayer_1_1/bias'variables/13/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUE:transformer_encoder_15/Encoder-FeedForwardLayer_2_1/kernel'variables/14/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUE8transformer_encoder_15/Encoder-FeedForwardLayer_2_1/bias'variables/15/.ATTRIBUTES/VARIABLE_VALUE*
�{
VARIABLE_VALUE@transformer_encoder_16/Encoder-SelfAttentionLayer-2/query/kernel'variables/16/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE>transformer_encoder_16/Encoder-SelfAttentionLayer-2/query/bias'variables/17/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE>transformer_encoder_16/Encoder-SelfAttentionLayer-2/key/kernel'variables/18/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUE<transformer_encoder_16/Encoder-SelfAttentionLayer-2/key/bias'variables/19/.ATTRIBUTES/VARIABLE_VALUE*
�{
VARIABLE_VALUE@transformer_encoder_16/Encoder-SelfAttentionLayer-2/value/kernel'variables/20/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE>transformer_encoder_16/Encoder-SelfAttentionLayer-2/value/bias'variables/21/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEKtransformer_encoder_16/Encoder-SelfAttentionLayer-2/attention_output/kernel'variables/22/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEItransformer_encoder_16/Encoder-SelfAttentionLayer-2/attention_output/bias'variables/23/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE=transformer_encoder_16/Encoder-1st-NormalizationLayer-2/gamma'variables/24/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUE<transformer_encoder_16/Encoder-1st-NormalizationLayer-2/beta'variables/25/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE=transformer_encoder_16/Encoder-2nd-NormalizationLayer-2/gamma'variables/26/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUE<transformer_encoder_16/Encoder-2nd-NormalizationLayer-2/beta'variables/27/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUE:transformer_encoder_16/Encoder-FeedForwardLayer_1_2/kernel'variables/28/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUE8transformer_encoder_16/Encoder-FeedForwardLayer_1_2/bias'variables/29/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUE:transformer_encoder_16/Encoder-FeedForwardLayer_2_2/kernel'variables/30/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUE8transformer_encoder_16/Encoder-FeedForwardLayer_2_2/bias'variables/31/.ATTRIBUTES/VARIABLE_VALUE*
�{
VARIABLE_VALUE@transformer_encoder_17/Encoder-SelfAttentionLayer-3/query/kernel'variables/32/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE>transformer_encoder_17/Encoder-SelfAttentionLayer-3/query/bias'variables/33/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE>transformer_encoder_17/Encoder-SelfAttentionLayer-3/key/kernel'variables/34/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUE<transformer_encoder_17/Encoder-SelfAttentionLayer-3/key/bias'variables/35/.ATTRIBUTES/VARIABLE_VALUE*
�{
VARIABLE_VALUE@transformer_encoder_17/Encoder-SelfAttentionLayer-3/value/kernel'variables/36/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE>transformer_encoder_17/Encoder-SelfAttentionLayer-3/value/bias'variables/37/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEKtransformer_encoder_17/Encoder-SelfAttentionLayer-3/attention_output/kernel'variables/38/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEItransformer_encoder_17/Encoder-SelfAttentionLayer-3/attention_output/bias'variables/39/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE=transformer_encoder_17/Encoder-1st-NormalizationLayer-3/gamma'variables/40/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUE<transformer_encoder_17/Encoder-1st-NormalizationLayer-3/beta'variables/41/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE=transformer_encoder_17/Encoder-2nd-NormalizationLayer-3/gamma'variables/42/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUE<transformer_encoder_17/Encoder-2nd-NormalizationLayer-3/beta'variables/43/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUE:transformer_encoder_17/Encoder-FeedForwardLayer_1_3/kernel'variables/44/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUE8transformer_encoder_17/Encoder-FeedForwardLayer_1_3/bias'variables/45/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUE:transformer_encoder_17/Encoder-FeedForwardLayer_2_3/kernel'variables/46/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUE8transformer_encoder_17/Encoder-FeedForwardLayer_2_3/bias'variables/47/.ATTRIBUTES/VARIABLE_VALUE*
* 
r
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
12
13
14*
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
$0
%1
&2
'3
(4
)5
*6
+7*
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
�kernel
	�bias*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�partial_output_shape
�full_output_shape
�kernel
	�bias*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�partial_output_shape
�full_output_shape
�kernel
	�bias*
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
20
31
42
53
64
75
86
97*
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
@0
A1
B2
C3
D4
E5
F6
G7*
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
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
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
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
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
�	variables
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
�layer_metrics
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
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
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
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
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
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
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
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameFinalLayerNorm/gammaFinalLayerNorm/beta#FullyConnectedLayerAreaRatio/kernel!FullyConnectedLayerAreaRatio/bias FullyConnectedLayerSolved/kernelFullyConnectedLayerSolved/biasPredictionAreaRatio/kernelPredictionAreaRatio/biasPredictionSolved/kernelPredictionSolved/bias@transformer_encoder_15/Encoder-SelfAttentionLayer-1/query/kernel>transformer_encoder_15/Encoder-SelfAttentionLayer-1/query/bias>transformer_encoder_15/Encoder-SelfAttentionLayer-1/key/kernel<transformer_encoder_15/Encoder-SelfAttentionLayer-1/key/bias@transformer_encoder_15/Encoder-SelfAttentionLayer-1/value/kernel>transformer_encoder_15/Encoder-SelfAttentionLayer-1/value/biasKtransformer_encoder_15/Encoder-SelfAttentionLayer-1/attention_output/kernelItransformer_encoder_15/Encoder-SelfAttentionLayer-1/attention_output/bias=transformer_encoder_15/Encoder-1st-NormalizationLayer-1/gamma<transformer_encoder_15/Encoder-1st-NormalizationLayer-1/beta=transformer_encoder_15/Encoder-2nd-NormalizationLayer-1/gamma<transformer_encoder_15/Encoder-2nd-NormalizationLayer-1/beta:transformer_encoder_15/Encoder-FeedForwardLayer_1_1/kernel8transformer_encoder_15/Encoder-FeedForwardLayer_1_1/bias:transformer_encoder_15/Encoder-FeedForwardLayer_2_1/kernel8transformer_encoder_15/Encoder-FeedForwardLayer_2_1/bias@transformer_encoder_16/Encoder-SelfAttentionLayer-2/query/kernel>transformer_encoder_16/Encoder-SelfAttentionLayer-2/query/bias>transformer_encoder_16/Encoder-SelfAttentionLayer-2/key/kernel<transformer_encoder_16/Encoder-SelfAttentionLayer-2/key/bias@transformer_encoder_16/Encoder-SelfAttentionLayer-2/value/kernel>transformer_encoder_16/Encoder-SelfAttentionLayer-2/value/biasKtransformer_encoder_16/Encoder-SelfAttentionLayer-2/attention_output/kernelItransformer_encoder_16/Encoder-SelfAttentionLayer-2/attention_output/bias=transformer_encoder_16/Encoder-1st-NormalizationLayer-2/gamma<transformer_encoder_16/Encoder-1st-NormalizationLayer-2/beta=transformer_encoder_16/Encoder-2nd-NormalizationLayer-2/gamma<transformer_encoder_16/Encoder-2nd-NormalizationLayer-2/beta:transformer_encoder_16/Encoder-FeedForwardLayer_1_2/kernel8transformer_encoder_16/Encoder-FeedForwardLayer_1_2/bias:transformer_encoder_16/Encoder-FeedForwardLayer_2_2/kernel8transformer_encoder_16/Encoder-FeedForwardLayer_2_2/bias@transformer_encoder_17/Encoder-SelfAttentionLayer-3/query/kernel>transformer_encoder_17/Encoder-SelfAttentionLayer-3/query/bias>transformer_encoder_17/Encoder-SelfAttentionLayer-3/key/kernel<transformer_encoder_17/Encoder-SelfAttentionLayer-3/key/bias@transformer_encoder_17/Encoder-SelfAttentionLayer-3/value/kernel>transformer_encoder_17/Encoder-SelfAttentionLayer-3/value/biasKtransformer_encoder_17/Encoder-SelfAttentionLayer-3/attention_output/kernelItransformer_encoder_17/Encoder-SelfAttentionLayer-3/attention_output/bias=transformer_encoder_17/Encoder-1st-NormalizationLayer-3/gamma<transformer_encoder_17/Encoder-1st-NormalizationLayer-3/beta=transformer_encoder_17/Encoder-2nd-NormalizationLayer-3/gamma<transformer_encoder_17/Encoder-2nd-NormalizationLayer-3/beta:transformer_encoder_17/Encoder-FeedForwardLayer_1_3/kernel8transformer_encoder_17/Encoder-FeedForwardLayer_1_3/bias:transformer_encoder_17/Encoder-FeedForwardLayer_2_3/kernel8transformer_encoder_17/Encoder-FeedForwardLayer_2_3/biasConst*G
Tin@
>2<*
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
GPU2*0J 8� *)
f$R"
 __inference__traced_save_1433327
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameFinalLayerNorm/gammaFinalLayerNorm/beta#FullyConnectedLayerAreaRatio/kernel!FullyConnectedLayerAreaRatio/bias FullyConnectedLayerSolved/kernelFullyConnectedLayerSolved/biasPredictionAreaRatio/kernelPredictionAreaRatio/biasPredictionSolved/kernelPredictionSolved/bias@transformer_encoder_15/Encoder-SelfAttentionLayer-1/query/kernel>transformer_encoder_15/Encoder-SelfAttentionLayer-1/query/bias>transformer_encoder_15/Encoder-SelfAttentionLayer-1/key/kernel<transformer_encoder_15/Encoder-SelfAttentionLayer-1/key/bias@transformer_encoder_15/Encoder-SelfAttentionLayer-1/value/kernel>transformer_encoder_15/Encoder-SelfAttentionLayer-1/value/biasKtransformer_encoder_15/Encoder-SelfAttentionLayer-1/attention_output/kernelItransformer_encoder_15/Encoder-SelfAttentionLayer-1/attention_output/bias=transformer_encoder_15/Encoder-1st-NormalizationLayer-1/gamma<transformer_encoder_15/Encoder-1st-NormalizationLayer-1/beta=transformer_encoder_15/Encoder-2nd-NormalizationLayer-1/gamma<transformer_encoder_15/Encoder-2nd-NormalizationLayer-1/beta:transformer_encoder_15/Encoder-FeedForwardLayer_1_1/kernel8transformer_encoder_15/Encoder-FeedForwardLayer_1_1/bias:transformer_encoder_15/Encoder-FeedForwardLayer_2_1/kernel8transformer_encoder_15/Encoder-FeedForwardLayer_2_1/bias@transformer_encoder_16/Encoder-SelfAttentionLayer-2/query/kernel>transformer_encoder_16/Encoder-SelfAttentionLayer-2/query/bias>transformer_encoder_16/Encoder-SelfAttentionLayer-2/key/kernel<transformer_encoder_16/Encoder-SelfAttentionLayer-2/key/bias@transformer_encoder_16/Encoder-SelfAttentionLayer-2/value/kernel>transformer_encoder_16/Encoder-SelfAttentionLayer-2/value/biasKtransformer_encoder_16/Encoder-SelfAttentionLayer-2/attention_output/kernelItransformer_encoder_16/Encoder-SelfAttentionLayer-2/attention_output/bias=transformer_encoder_16/Encoder-1st-NormalizationLayer-2/gamma<transformer_encoder_16/Encoder-1st-NormalizationLayer-2/beta=transformer_encoder_16/Encoder-2nd-NormalizationLayer-2/gamma<transformer_encoder_16/Encoder-2nd-NormalizationLayer-2/beta:transformer_encoder_16/Encoder-FeedForwardLayer_1_2/kernel8transformer_encoder_16/Encoder-FeedForwardLayer_1_2/bias:transformer_encoder_16/Encoder-FeedForwardLayer_2_2/kernel8transformer_encoder_16/Encoder-FeedForwardLayer_2_2/bias@transformer_encoder_17/Encoder-SelfAttentionLayer-3/query/kernel>transformer_encoder_17/Encoder-SelfAttentionLayer-3/query/bias>transformer_encoder_17/Encoder-SelfAttentionLayer-3/key/kernel<transformer_encoder_17/Encoder-SelfAttentionLayer-3/key/bias@transformer_encoder_17/Encoder-SelfAttentionLayer-3/value/kernel>transformer_encoder_17/Encoder-SelfAttentionLayer-3/value/biasKtransformer_encoder_17/Encoder-SelfAttentionLayer-3/attention_output/kernelItransformer_encoder_17/Encoder-SelfAttentionLayer-3/attention_output/bias=transformer_encoder_17/Encoder-1st-NormalizationLayer-3/gamma<transformer_encoder_17/Encoder-1st-NormalizationLayer-3/beta=transformer_encoder_17/Encoder-2nd-NormalizationLayer-3/gamma<transformer_encoder_17/Encoder-2nd-NormalizationLayer-3/beta:transformer_encoder_17/Encoder-FeedForwardLayer_1_3/kernel8transformer_encoder_17/Encoder-FeedForwardLayer_1_3/bias:transformer_encoder_17/Encoder-FeedForwardLayer_2_3/kernel8transformer_encoder_17/Encoder-FeedForwardLayer_2_3/bias*F
Tin?
=2;*
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
GPU2*0J 8� *,
f'R%
#__inference__traced_restore_1433510��/
�
�
0__inference_FinalLayerNorm_layer_call_fn_1432734

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
GPU2*0J 8� *T
fORM
K__inference_FinalLayerNorm_layer_call_and_return_conditional_losses_1429890s
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
StatefulPartitionedCallStatefulPartitionedCall:'#
!
_user_specified_name	1432730:'#
!
_user_specified_name	1432728:S O
+
_output_shapes
:���������P	
 
_user_specified_nameinputs
�
e
I__inference_MaskingLayer_layer_call_and_return_conditional_losses_1429180

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
S__inference_transformer_encoder_16_layer_call_and_return_conditional_losses_1432111

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
:���������P	]
dropout_16/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?�
dropout_16/dropout/MulMul-Encoder-FeedForwardLayer_2_2/BiasAdd:output:0!dropout_16/dropout/Const:output:0*
T0*+
_output_shapes
:���������P	�
dropout_16/dropout/ShapeShape-Encoder-FeedForwardLayer_2_2/BiasAdd:output:0*
T0*
_output_shapes
::���
/dropout_16/dropout/random_uniform/RandomUniformRandomUniform!dropout_16/dropout/Shape:output:0*
T0*+
_output_shapes
:���������P	*
dtype0*
seed2*
seed��f
!dropout_16/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=�
dropout_16/dropout/GreaterEqualGreaterEqual8dropout_16/dropout/random_uniform/RandomUniform:output:0*dropout_16/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:���������P	_
dropout_16/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout_16/dropout/SelectV2SelectV2#dropout_16/dropout/GreaterEqual:z:0dropout_16/dropout/Mul:z:0#dropout_16/dropout/Const_1:output:0*
T0*+
_output_shapes
:���������P	�
Encoder-2nd-AdditionLayer-2/addAddV2#Encoder-1st-AdditionLayer-2/add:z:0$dropout_16/dropout/SelectV2:output:0*
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
�
^
2__inference_ConcatenateLayer_layer_call_fn_1432834
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
GPU2*0J 8� *V
fQRO
M__inference_ConcatenateLayer_layer_call_and_return_conditional_losses_1429921`
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
�
y
]__inference_ReduceStackDimensionViaSummation_layer_call_and_return_conditional_losses_1432802

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
�
�
8__inference_transformer_encoder_15_layer_call_fn_1431483

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
GPU2*0J 8� *\
fWRU
S__inference_transformer_encoder_15_layer_call_and_return_conditional_losses_1430190s
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
StatefulPartitionedCallStatefulPartitionedCall:'#
!
_user_specified_name	1431477:'#
!
_user_specified_name	1431475:'#
!
_user_specified_name	1431473:'#
!
_user_specified_name	1431471:'#
!
_user_specified_name	1431469:'#
!
_user_specified_name	1431467:'
#
!
_user_specified_name	1431465:'	#
!
_user_specified_name	1431463:'#
!
_user_specified_name	1431461:'#
!
_user_specified_name	1431459:'#
!
_user_specified_name	1431457:'#
!
_user_specified_name	1431455:'#
!
_user_specified_name	1431453:'#
!
_user_specified_name	1431451:'#
!
_user_specified_name	1431449:'#
!
_user_specified_name	1431447:S O
+
_output_shapes
:���������P	
 
_user_specified_nameinputs
�
J
.__inference_MaskingLayer_layer_call_fn_1431394

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
GPU2*0J 8� *R
fMRK
I__inference_MaskingLayer_layer_call_and_return_conditional_losses_1429180d
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
��
�5
#__inference__traced_restore_1433510
file_prefix3
%assignvariableop_finallayernorm_gamma:	4
&assignvariableop_1_finallayernorm_beta:	H
6assignvariableop_2_fullyconnectedlayerarearatio_kernel:

B
4assignvariableop_3_fullyconnectedlayerarearatio_bias:
E
3assignvariableop_4_fullyconnectedlayersolved_kernel:

?
1assignvariableop_5_fullyconnectedlayersolved_bias:
?
-assignvariableop_6_predictionarearatio_kernel:
9
+assignvariableop_7_predictionarearatio_bias:<
*assignvariableop_8_predictionsolved_kernel:
6
(assignvariableop_9_predictionsolved_bias:j
Tassignvariableop_10_transformer_encoder_15_encoder_selfattentionlayer_1_query_kernel:	d
Rassignvariableop_11_transformer_encoder_15_encoder_selfattentionlayer_1_query_bias:h
Rassignvariableop_12_transformer_encoder_15_encoder_selfattentionlayer_1_key_kernel:	b
Passignvariableop_13_transformer_encoder_15_encoder_selfattentionlayer_1_key_bias:j
Tassignvariableop_14_transformer_encoder_15_encoder_selfattentionlayer_1_value_kernel:	d
Rassignvariableop_15_transformer_encoder_15_encoder_selfattentionlayer_1_value_bias:u
_assignvariableop_16_transformer_encoder_15_encoder_selfattentionlayer_1_attention_output_kernel:	k
]assignvariableop_17_transformer_encoder_15_encoder_selfattentionlayer_1_attention_output_bias:	_
Qassignvariableop_18_transformer_encoder_15_encoder_1st_normalizationlayer_1_gamma:	^
Passignvariableop_19_transformer_encoder_15_encoder_1st_normalizationlayer_1_beta:	_
Qassignvariableop_20_transformer_encoder_15_encoder_2nd_normalizationlayer_1_gamma:	^
Passignvariableop_21_transformer_encoder_15_encoder_2nd_normalizationlayer_1_beta:	`
Nassignvariableop_22_transformer_encoder_15_encoder_feedforwardlayer_1_1_kernel:		Z
Lassignvariableop_23_transformer_encoder_15_encoder_feedforwardlayer_1_1_bias:	`
Nassignvariableop_24_transformer_encoder_15_encoder_feedforwardlayer_2_1_kernel:		Z
Lassignvariableop_25_transformer_encoder_15_encoder_feedforwardlayer_2_1_bias:	j
Tassignvariableop_26_transformer_encoder_16_encoder_selfattentionlayer_2_query_kernel:	d
Rassignvariableop_27_transformer_encoder_16_encoder_selfattentionlayer_2_query_bias:h
Rassignvariableop_28_transformer_encoder_16_encoder_selfattentionlayer_2_key_kernel:	b
Passignvariableop_29_transformer_encoder_16_encoder_selfattentionlayer_2_key_bias:j
Tassignvariableop_30_transformer_encoder_16_encoder_selfattentionlayer_2_value_kernel:	d
Rassignvariableop_31_transformer_encoder_16_encoder_selfattentionlayer_2_value_bias:u
_assignvariableop_32_transformer_encoder_16_encoder_selfattentionlayer_2_attention_output_kernel:	k
]assignvariableop_33_transformer_encoder_16_encoder_selfattentionlayer_2_attention_output_bias:	_
Qassignvariableop_34_transformer_encoder_16_encoder_1st_normalizationlayer_2_gamma:	^
Passignvariableop_35_transformer_encoder_16_encoder_1st_normalizationlayer_2_beta:	_
Qassignvariableop_36_transformer_encoder_16_encoder_2nd_normalizationlayer_2_gamma:	^
Passignvariableop_37_transformer_encoder_16_encoder_2nd_normalizationlayer_2_beta:	`
Nassignvariableop_38_transformer_encoder_16_encoder_feedforwardlayer_1_2_kernel:		Z
Lassignvariableop_39_transformer_encoder_16_encoder_feedforwardlayer_1_2_bias:	`
Nassignvariableop_40_transformer_encoder_16_encoder_feedforwardlayer_2_2_kernel:		Z
Lassignvariableop_41_transformer_encoder_16_encoder_feedforwardlayer_2_2_bias:	j
Tassignvariableop_42_transformer_encoder_17_encoder_selfattentionlayer_3_query_kernel:	d
Rassignvariableop_43_transformer_encoder_17_encoder_selfattentionlayer_3_query_bias:h
Rassignvariableop_44_transformer_encoder_17_encoder_selfattentionlayer_3_key_kernel:	b
Passignvariableop_45_transformer_encoder_17_encoder_selfattentionlayer_3_key_bias:j
Tassignvariableop_46_transformer_encoder_17_encoder_selfattentionlayer_3_value_kernel:	d
Rassignvariableop_47_transformer_encoder_17_encoder_selfattentionlayer_3_value_bias:u
_assignvariableop_48_transformer_encoder_17_encoder_selfattentionlayer_3_attention_output_kernel:	k
]assignvariableop_49_transformer_encoder_17_encoder_selfattentionlayer_3_attention_output_bias:	_
Qassignvariableop_50_transformer_encoder_17_encoder_1st_normalizationlayer_3_gamma:	^
Passignvariableop_51_transformer_encoder_17_encoder_1st_normalizationlayer_3_beta:	_
Qassignvariableop_52_transformer_encoder_17_encoder_2nd_normalizationlayer_3_gamma:	^
Passignvariableop_53_transformer_encoder_17_encoder_2nd_normalizationlayer_3_beta:	`
Nassignvariableop_54_transformer_encoder_17_encoder_feedforwardlayer_1_3_kernel:		Z
Lassignvariableop_55_transformer_encoder_17_encoder_feedforwardlayer_1_3_bias:	`
Nassignvariableop_56_transformer_encoder_17_encoder_feedforwardlayer_2_3_kernel:		Z
Lassignvariableop_57_transformer_encoder_17_encoder_feedforwardlayer_2_3_bias:	
identity_59��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_31�AssignVariableOp_32�AssignVariableOp_33�AssignVariableOp_34�AssignVariableOp_35�AssignVariableOp_36�AssignVariableOp_37�AssignVariableOp_38�AssignVariableOp_39�AssignVariableOp_4�AssignVariableOp_40�AssignVariableOp_41�AssignVariableOp_42�AssignVariableOp_43�AssignVariableOp_44�AssignVariableOp_45�AssignVariableOp_46�AssignVariableOp_47�AssignVariableOp_48�AssignVariableOp_49�AssignVariableOp_5�AssignVariableOp_50�AssignVariableOp_51�AssignVariableOp_52�AssignVariableOp_53�AssignVariableOp_54�AssignVariableOp_55�AssignVariableOp_56�AssignVariableOp_57�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:;*
dtype0*�
value�B�;B5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB'variables/26/.ATTRIBUTES/VARIABLE_VALUEB'variables/27/.ATTRIBUTES/VARIABLE_VALUEB'variables/28/.ATTRIBUTES/VARIABLE_VALUEB'variables/29/.ATTRIBUTES/VARIABLE_VALUEB'variables/30/.ATTRIBUTES/VARIABLE_VALUEB'variables/31/.ATTRIBUTES/VARIABLE_VALUEB'variables/32/.ATTRIBUTES/VARIABLE_VALUEB'variables/33/.ATTRIBUTES/VARIABLE_VALUEB'variables/34/.ATTRIBUTES/VARIABLE_VALUEB'variables/35/.ATTRIBUTES/VARIABLE_VALUEB'variables/36/.ATTRIBUTES/VARIABLE_VALUEB'variables/37/.ATTRIBUTES/VARIABLE_VALUEB'variables/38/.ATTRIBUTES/VARIABLE_VALUEB'variables/39/.ATTRIBUTES/VARIABLE_VALUEB'variables/40/.ATTRIBUTES/VARIABLE_VALUEB'variables/41/.ATTRIBUTES/VARIABLE_VALUEB'variables/42/.ATTRIBUTES/VARIABLE_VALUEB'variables/43/.ATTRIBUTES/VARIABLE_VALUEB'variables/44/.ATTRIBUTES/VARIABLE_VALUEB'variables/45/.ATTRIBUTES/VARIABLE_VALUEB'variables/46/.ATTRIBUTES/VARIABLE_VALUEB'variables/47/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:;*
dtype0*�
value�B~;B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes�
�:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*I
dtypes?
=2;[
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
AssignVariableOp_2AssignVariableOp6assignvariableop_2_fullyconnectedlayerarearatio_kernelIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOp4assignvariableop_3_fullyconnectedlayerarearatio_biasIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOp3assignvariableop_4_fullyconnectedlayersolved_kernelIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOp1assignvariableop_5_fullyconnectedlayersolved_biasIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOp-assignvariableop_6_predictionarearatio_kernelIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOp+assignvariableop_7_predictionarearatio_biasIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOp*assignvariableop_8_predictionsolved_kernelIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOp(assignvariableop_9_predictionsolved_biasIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOpTassignvariableop_10_transformer_encoder_15_encoder_selfattentionlayer_1_query_kernelIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOpRassignvariableop_11_transformer_encoder_15_encoder_selfattentionlayer_1_query_biasIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOpRassignvariableop_12_transformer_encoder_15_encoder_selfattentionlayer_1_key_kernelIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOpPassignvariableop_13_transformer_encoder_15_encoder_selfattentionlayer_1_key_biasIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOpTassignvariableop_14_transformer_encoder_15_encoder_selfattentionlayer_1_value_kernelIdentity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOpRassignvariableop_15_transformer_encoder_15_encoder_selfattentionlayer_1_value_biasIdentity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOp_assignvariableop_16_transformer_encoder_15_encoder_selfattentionlayer_1_attention_output_kernelIdentity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOp]assignvariableop_17_transformer_encoder_15_encoder_selfattentionlayer_1_attention_output_biasIdentity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_18AssignVariableOpQassignvariableop_18_transformer_encoder_15_encoder_1st_normalizationlayer_1_gammaIdentity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOpPassignvariableop_19_transformer_encoder_15_encoder_1st_normalizationlayer_1_betaIdentity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_20AssignVariableOpQassignvariableop_20_transformer_encoder_15_encoder_2nd_normalizationlayer_1_gammaIdentity_20:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_21AssignVariableOpPassignvariableop_21_transformer_encoder_15_encoder_2nd_normalizationlayer_1_betaIdentity_21:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_22AssignVariableOpNassignvariableop_22_transformer_encoder_15_encoder_feedforwardlayer_1_1_kernelIdentity_22:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_23AssignVariableOpLassignvariableop_23_transformer_encoder_15_encoder_feedforwardlayer_1_1_biasIdentity_23:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_24AssignVariableOpNassignvariableop_24_transformer_encoder_15_encoder_feedforwardlayer_2_1_kernelIdentity_24:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_25AssignVariableOpLassignvariableop_25_transformer_encoder_15_encoder_feedforwardlayer_2_1_biasIdentity_25:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_26AssignVariableOpTassignvariableop_26_transformer_encoder_16_encoder_selfattentionlayer_2_query_kernelIdentity_26:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_27AssignVariableOpRassignvariableop_27_transformer_encoder_16_encoder_selfattentionlayer_2_query_biasIdentity_27:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_28AssignVariableOpRassignvariableop_28_transformer_encoder_16_encoder_selfattentionlayer_2_key_kernelIdentity_28:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_29AssignVariableOpPassignvariableop_29_transformer_encoder_16_encoder_selfattentionlayer_2_key_biasIdentity_29:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_30AssignVariableOpTassignvariableop_30_transformer_encoder_16_encoder_selfattentionlayer_2_value_kernelIdentity_30:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_31AssignVariableOpRassignvariableop_31_transformer_encoder_16_encoder_selfattentionlayer_2_value_biasIdentity_31:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_32AssignVariableOp_assignvariableop_32_transformer_encoder_16_encoder_selfattentionlayer_2_attention_output_kernelIdentity_32:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_33AssignVariableOp]assignvariableop_33_transformer_encoder_16_encoder_selfattentionlayer_2_attention_output_biasIdentity_33:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_34AssignVariableOpQassignvariableop_34_transformer_encoder_16_encoder_1st_normalizationlayer_2_gammaIdentity_34:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_35AssignVariableOpPassignvariableop_35_transformer_encoder_16_encoder_1st_normalizationlayer_2_betaIdentity_35:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_36AssignVariableOpQassignvariableop_36_transformer_encoder_16_encoder_2nd_normalizationlayer_2_gammaIdentity_36:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_37AssignVariableOpPassignvariableop_37_transformer_encoder_16_encoder_2nd_normalizationlayer_2_betaIdentity_37:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_38AssignVariableOpNassignvariableop_38_transformer_encoder_16_encoder_feedforwardlayer_1_2_kernelIdentity_38:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_39AssignVariableOpLassignvariableop_39_transformer_encoder_16_encoder_feedforwardlayer_1_2_biasIdentity_39:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_40AssignVariableOpNassignvariableop_40_transformer_encoder_16_encoder_feedforwardlayer_2_2_kernelIdentity_40:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_41AssignVariableOpLassignvariableop_41_transformer_encoder_16_encoder_feedforwardlayer_2_2_biasIdentity_41:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_42AssignVariableOpTassignvariableop_42_transformer_encoder_17_encoder_selfattentionlayer_3_query_kernelIdentity_42:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_43AssignVariableOpRassignvariableop_43_transformer_encoder_17_encoder_selfattentionlayer_3_query_biasIdentity_43:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_44AssignVariableOpRassignvariableop_44_transformer_encoder_17_encoder_selfattentionlayer_3_key_kernelIdentity_44:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_45AssignVariableOpPassignvariableop_45_transformer_encoder_17_encoder_selfattentionlayer_3_key_biasIdentity_45:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_46AssignVariableOpTassignvariableop_46_transformer_encoder_17_encoder_selfattentionlayer_3_value_kernelIdentity_46:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_47AssignVariableOpRassignvariableop_47_transformer_encoder_17_encoder_selfattentionlayer_3_value_biasIdentity_47:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_48AssignVariableOp_assignvariableop_48_transformer_encoder_17_encoder_selfattentionlayer_3_attention_output_kernelIdentity_48:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_49AssignVariableOp]assignvariableop_49_transformer_encoder_17_encoder_selfattentionlayer_3_attention_output_biasIdentity_49:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_50AssignVariableOpQassignvariableop_50_transformer_encoder_17_encoder_1st_normalizationlayer_3_gammaIdentity_50:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_51AssignVariableOpPassignvariableop_51_transformer_encoder_17_encoder_1st_normalizationlayer_3_betaIdentity_51:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_52AssignVariableOpQassignvariableop_52_transformer_encoder_17_encoder_2nd_normalizationlayer_3_gammaIdentity_52:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_53AssignVariableOpPassignvariableop_53_transformer_encoder_17_encoder_2nd_normalizationlayer_3_betaIdentity_53:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_54AssignVariableOpNassignvariableop_54_transformer_encoder_17_encoder_feedforwardlayer_1_3_kernelIdentity_54:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_55AssignVariableOpLassignvariableop_55_transformer_encoder_17_encoder_feedforwardlayer_1_3_biasIdentity_55:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_56AssignVariableOpNassignvariableop_56_transformer_encoder_17_encoder_feedforwardlayer_2_3_kernelIdentity_56:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_57AssignVariableOpLassignvariableop_57_transformer_encoder_17_encoder_feedforwardlayer_2_3_biasIdentity_57:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 �

Identity_58Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_59IdentityIdentity_58:output:0^NoOp_1*
T0*
_output_shapes
: �

NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
_output_shapes
 "#
identity_59Identity_59:output:0*(
_construction_contextkEagerRuntime*�
_input_shapesx
v: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2*
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
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92$
AssignVariableOpAssignVariableOp:X:T
R
_user_specified_name:8transformer_encoder_17/Encoder-FeedForwardLayer_2_3/bias:Z9V
T
_user_specified_name<:transformer_encoder_17/Encoder-FeedForwardLayer_2_3/kernel:X8T
R
_user_specified_name:8transformer_encoder_17/Encoder-FeedForwardLayer_1_3/bias:Z7V
T
_user_specified_name<:transformer_encoder_17/Encoder-FeedForwardLayer_1_3/kernel:\6X
V
_user_specified_name><transformer_encoder_17/Encoder-2nd-NormalizationLayer-3/beta:]5Y
W
_user_specified_name?=transformer_encoder_17/Encoder-2nd-NormalizationLayer-3/gamma:\4X
V
_user_specified_name><transformer_encoder_17/Encoder-1st-NormalizationLayer-3/beta:]3Y
W
_user_specified_name?=transformer_encoder_17/Encoder-1st-NormalizationLayer-3/gamma:i2e
c
_user_specified_nameKItransformer_encoder_17/Encoder-SelfAttentionLayer-3/attention_output/bias:k1g
e
_user_specified_nameMKtransformer_encoder_17/Encoder-SelfAttentionLayer-3/attention_output/kernel:^0Z
X
_user_specified_name@>transformer_encoder_17/Encoder-SelfAttentionLayer-3/value/bias:`/\
Z
_user_specified_nameB@transformer_encoder_17/Encoder-SelfAttentionLayer-3/value/kernel:\.X
V
_user_specified_name><transformer_encoder_17/Encoder-SelfAttentionLayer-3/key/bias:^-Z
X
_user_specified_name@>transformer_encoder_17/Encoder-SelfAttentionLayer-3/key/kernel:^,Z
X
_user_specified_name@>transformer_encoder_17/Encoder-SelfAttentionLayer-3/query/bias:`+\
Z
_user_specified_nameB@transformer_encoder_17/Encoder-SelfAttentionLayer-3/query/kernel:X*T
R
_user_specified_name:8transformer_encoder_16/Encoder-FeedForwardLayer_2_2/bias:Z)V
T
_user_specified_name<:transformer_encoder_16/Encoder-FeedForwardLayer_2_2/kernel:X(T
R
_user_specified_name:8transformer_encoder_16/Encoder-FeedForwardLayer_1_2/bias:Z'V
T
_user_specified_name<:transformer_encoder_16/Encoder-FeedForwardLayer_1_2/kernel:\&X
V
_user_specified_name><transformer_encoder_16/Encoder-2nd-NormalizationLayer-2/beta:]%Y
W
_user_specified_name?=transformer_encoder_16/Encoder-2nd-NormalizationLayer-2/gamma:\$X
V
_user_specified_name><transformer_encoder_16/Encoder-1st-NormalizationLayer-2/beta:]#Y
W
_user_specified_name?=transformer_encoder_16/Encoder-1st-NormalizationLayer-2/gamma:i"e
c
_user_specified_nameKItransformer_encoder_16/Encoder-SelfAttentionLayer-2/attention_output/bias:k!g
e
_user_specified_nameMKtransformer_encoder_16/Encoder-SelfAttentionLayer-2/attention_output/kernel:^ Z
X
_user_specified_name@>transformer_encoder_16/Encoder-SelfAttentionLayer-2/value/bias:`\
Z
_user_specified_nameB@transformer_encoder_16/Encoder-SelfAttentionLayer-2/value/kernel:\X
V
_user_specified_name><transformer_encoder_16/Encoder-SelfAttentionLayer-2/key/bias:^Z
X
_user_specified_name@>transformer_encoder_16/Encoder-SelfAttentionLayer-2/key/kernel:^Z
X
_user_specified_name@>transformer_encoder_16/Encoder-SelfAttentionLayer-2/query/bias:`\
Z
_user_specified_nameB@transformer_encoder_16/Encoder-SelfAttentionLayer-2/query/kernel:XT
R
_user_specified_name:8transformer_encoder_15/Encoder-FeedForwardLayer_2_1/bias:ZV
T
_user_specified_name<:transformer_encoder_15/Encoder-FeedForwardLayer_2_1/kernel:XT
R
_user_specified_name:8transformer_encoder_15/Encoder-FeedForwardLayer_1_1/bias:ZV
T
_user_specified_name<:transformer_encoder_15/Encoder-FeedForwardLayer_1_1/kernel:\X
V
_user_specified_name><transformer_encoder_15/Encoder-2nd-NormalizationLayer-1/beta:]Y
W
_user_specified_name?=transformer_encoder_15/Encoder-2nd-NormalizationLayer-1/gamma:\X
V
_user_specified_name><transformer_encoder_15/Encoder-1st-NormalizationLayer-1/beta:]Y
W
_user_specified_name?=transformer_encoder_15/Encoder-1st-NormalizationLayer-1/gamma:ie
c
_user_specified_nameKItransformer_encoder_15/Encoder-SelfAttentionLayer-1/attention_output/bias:kg
e
_user_specified_nameMKtransformer_encoder_15/Encoder-SelfAttentionLayer-1/attention_output/kernel:^Z
X
_user_specified_name@>transformer_encoder_15/Encoder-SelfAttentionLayer-1/value/bias:`\
Z
_user_specified_nameB@transformer_encoder_15/Encoder-SelfAttentionLayer-1/value/kernel:\X
V
_user_specified_name><transformer_encoder_15/Encoder-SelfAttentionLayer-1/key/bias:^Z
X
_user_specified_name@>transformer_encoder_15/Encoder-SelfAttentionLayer-1/key/kernel:^Z
X
_user_specified_name@>transformer_encoder_15/Encoder-SelfAttentionLayer-1/query/bias:`\
Z
_user_specified_nameB@transformer_encoder_15/Encoder-SelfAttentionLayer-1/query/kernel:5
1
/
_user_specified_namePredictionSolved/bias:7	3
1
_user_specified_namePredictionSolved/kernel:84
2
_user_specified_namePredictionAreaRatio/bias::6
4
_user_specified_namePredictionAreaRatio/kernel:>:
8
_user_specified_name FullyConnectedLayerSolved/bias:@<
:
_user_specified_name" FullyConnectedLayerSolved/kernel:A=
;
_user_specified_name#!FullyConnectedLayerAreaRatio/bias:C?
=
_user_specified_name%#FullyConnectedLayerAreaRatio/kernel:3/
-
_user_specified_nameFinalLayerNorm/beta:40
.
_user_specified_nameFinalLayerNorm/gamma:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
_
C__inference_Output_layer_call_and_return_conditional_losses_1432955

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
P__inference_PredictionAreaRatio_layer_call_and_return_conditional_losses_1432915

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
�
D
(__inference_Output_layer_call_fn_1432945

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
GPU2*0J 8� *L
fGRE
C__inference_Output_layer_call_and_return_conditional_losses_1430691Q
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
�
y
]__inference_ReduceStackDimensionViaSummation_layer_call_and_return_conditional_losses_1430653

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
�
�
8__inference_transformer_encoder_17_layer_call_fn_1432363

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
GPU2*0J 8� *\
fWRU
S__inference_transformer_encoder_17_layer_call_and_return_conditional_losses_1430606s
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
StatefulPartitionedCallStatefulPartitionedCall:'#
!
_user_specified_name	1432357:'#
!
_user_specified_name	1432355:'#
!
_user_specified_name	1432353:'#
!
_user_specified_name	1432351:'#
!
_user_specified_name	1432349:'#
!
_user_specified_name	1432347:'
#
!
_user_specified_name	1432345:'	#
!
_user_specified_name	1432343:'#
!
_user_specified_name	1432341:'#
!
_user_specified_name	1432339:'#
!
_user_specified_name	1432337:'#
!
_user_specified_name	1432335:'#
!
_user_specified_name	1432333:'#
!
_user_specified_name	1432331:'#
!
_user_specified_name	1432329:'#
!
_user_specified_name	1432327:S O
+
_output_shapes
:���������P	
 
_user_specified_nameinputs
�
^
B__inference_ReduceStackDimensionViaSummation_layer_call_fn_1432786

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
GPU2*0J 8� *f
faR_
]__inference_ReduceStackDimensionViaSummation_layer_call_and_return_conditional_losses_1430653`
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
�
R
6__inference_StandardizeTimeLimit_layer_call_fn_1432812

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
GPU2*0J 8� *Z
fURS
Q__inference_StandardizeTimeLimit_layer_call_and_return_conditional_losses_1430663`
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
�
�
V__inference_FullyConnectedLayerSolved_layer_call_and_return_conditional_losses_1429941

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
�
D
(__inference_Output_layer_call_fn_1432940

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
GPU2*0J 8� *L
fGRE
C__inference_Output_layer_call_and_return_conditional_losses_1430006Q
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
��
�
S__inference_transformer_encoder_16_layer_call_and_return_conditional_losses_1430398

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
dropout_16/IdentityIdentity-Encoder-FeedForwardLayer_2_2/BiasAdd:output:0*
T0*+
_output_shapes
:���������P	�
Encoder-2nd-AdditionLayer-2/addAddV2#Encoder-1st-AdditionLayer-2/add:z:0dropout_16/Identity:output:0*
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
M__inference_PredictionSolved_layer_call_and_return_conditional_losses_1432935

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
S__inference_transformer_encoder_17_layer_call_and_return_conditional_losses_1429814

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
:���������P	]
dropout_17/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?�
dropout_17/dropout/MulMul-Encoder-FeedForwardLayer_2_3/BiasAdd:output:0!dropout_17/dropout/Const:output:0*
T0*+
_output_shapes
:���������P	�
dropout_17/dropout/ShapeShape-Encoder-FeedForwardLayer_2_3/BiasAdd:output:0*
T0*
_output_shapes
::���
/dropout_17/dropout/random_uniform/RandomUniformRandomUniform!dropout_17/dropout/Shape:output:0*
T0*+
_output_shapes
:���������P	*
dtype0*
seed2*
seed��f
!dropout_17/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=�
dropout_17/dropout/GreaterEqualGreaterEqual8dropout_17/dropout/random_uniform/RandomUniform:output:0*dropout_17/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:���������P	_
dropout_17/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout_17/dropout/SelectV2SelectV2#dropout_17/dropout/GreaterEqual:z:0dropout_17/dropout/Mul:z:0#dropout_17/dropout/Const_1:output:0*
T0*+
_output_shapes
:���������P	�
Encoder-2nd-AdditionLayer-3/addAddV2#Encoder-1st-AdditionLayer-3/add:z:0$dropout_17/dropout/SelectV2:output:0*
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
�
m
Q__inference_StandardizeTimeLimit_layer_call_and_return_conditional_losses_1432828

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
S__inference_transformer_encoder_15_layer_call_and_return_conditional_losses_1431845

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
dropout_15/IdentityIdentity-Encoder-FeedForwardLayer_2_1/BiasAdd:output:0*
T0*+
_output_shapes
:���������P	�
Encoder-2nd-AdditionLayer-1/addAddV2#Encoder-1st-AdditionLayer-1/add:z:0dropout_15/Identity:output:0*
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
�
�
5__inference_PredictionAreaRatio_layer_call_fn_1432904

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
GPU2*0J 8� *Y
fTRR
P__inference_PredictionAreaRatio_layer_call_and_return_conditional_losses_1429996o
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
StatefulPartitionedCallStatefulPartitionedCall:'#
!
_user_specified_name	1432900:'#
!
_user_specified_name	1432898:O K
'
_output_shapes
:���������

 
_user_specified_nameinputs
�

�
P__inference_PredictionAreaRatio_layer_call_and_return_conditional_losses_1429996

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
S__inference_transformer_encoder_16_layer_call_and_return_conditional_losses_1429592

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
:���������P	]
dropout_16/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?�
dropout_16/dropout/MulMul-Encoder-FeedForwardLayer_2_2/BiasAdd:output:0!dropout_16/dropout/Const:output:0*
T0*+
_output_shapes
:���������P	�
dropout_16/dropout/ShapeShape-Encoder-FeedForwardLayer_2_2/BiasAdd:output:0*
T0*
_output_shapes
::���
/dropout_16/dropout/random_uniform/RandomUniformRandomUniform!dropout_16/dropout/Shape:output:0*
T0*+
_output_shapes
:���������P	*
dtype0*
seed2*
seed��f
!dropout_16/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=�
dropout_16/dropout/GreaterEqualGreaterEqual8dropout_16/dropout/random_uniform/RandomUniform:output:0*dropout_16/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:���������P	_
dropout_16/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout_16/dropout/SelectV2SelectV2#dropout_16/dropout/GreaterEqual:z:0dropout_16/dropout/Mul:z:0#dropout_16/dropout/Const_1:output:0*
T0*+
_output_shapes
:���������P	�
Encoder-2nd-AdditionLayer-2/addAddV2#Encoder-1st-AdditionLayer-2/add:z:0$dropout_16/dropout/SelectV2:output:0*
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
�/
�
)__inference_model_5_layer_call_fn_1430820
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



unknown_52:


unknown_53:


unknown_54:

unknown_55:


unknown_56:
identity

identity_1��StatefulPartitionedCall�
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
unknown_52
unknown_53
unknown_54
unknown_55
unknown_56*G
Tin@
>2<*
Tout
2*
_collective_manager_ids
 *
_output_shapes

::*\
_read_only_resource_inputs>
<:	
 !"#$%&'()*+,-./0123456789:;*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_model_5_layer_call_and_return_conditional_losses_1430011`
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes
:b

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*
_output_shapes
:<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:���������P	:���������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:';#
!
_user_specified_name	1430814:':#
!
_user_specified_name	1430812:'9#
!
_user_specified_name	1430810:'8#
!
_user_specified_name	1430808:'7#
!
_user_specified_name	1430806:'6#
!
_user_specified_name	1430804:'5#
!
_user_specified_name	1430802:'4#
!
_user_specified_name	1430800:'3#
!
_user_specified_name	1430798:'2#
!
_user_specified_name	1430796:'1#
!
_user_specified_name	1430794:'0#
!
_user_specified_name	1430792:'/#
!
_user_specified_name	1430790:'.#
!
_user_specified_name	1430788:'-#
!
_user_specified_name	1430786:',#
!
_user_specified_name	1430784:'+#
!
_user_specified_name	1430782:'*#
!
_user_specified_name	1430780:')#
!
_user_specified_name	1430778:'(#
!
_user_specified_name	1430776:''#
!
_user_specified_name	1430774:'&#
!
_user_specified_name	1430772:'%#
!
_user_specified_name	1430770:'$#
!
_user_specified_name	1430768:'##
!
_user_specified_name	1430766:'"#
!
_user_specified_name	1430764:'!#
!
_user_specified_name	1430762:' #
!
_user_specified_name	1430760:'#
!
_user_specified_name	1430758:'#
!
_user_specified_name	1430756:'#
!
_user_specified_name	1430754:'#
!
_user_specified_name	1430752:'#
!
_user_specified_name	1430750:'#
!
_user_specified_name	1430748:'#
!
_user_specified_name	1430746:'#
!
_user_specified_name	1430744:'#
!
_user_specified_name	1430742:'#
!
_user_specified_name	1430740:'#
!
_user_specified_name	1430738:'#
!
_user_specified_name	1430736:'#
!
_user_specified_name	1430734:'#
!
_user_specified_name	1430732:'#
!
_user_specified_name	1430730:'#
!
_user_specified_name	1430728:'#
!
_user_specified_name	1430726:'#
!
_user_specified_name	1430724:'#
!
_user_specified_name	1430722:'#
!
_user_specified_name	1430720:'#
!
_user_specified_name	1430718:'
#
!
_user_specified_name	1430716:'	#
!
_user_specified_name	1430714:'#
!
_user_specified_name	1430712:'#
!
_user_specified_name	1430710:'#
!
_user_specified_name	1430708:'#
!
_user_specified_name	1430706:'#
!
_user_specified_name	1430704:'#
!
_user_specified_name	1430702:'#
!
_user_specified_name	1430700:WS
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
�
�
8__inference_transformer_encoder_16_layer_call_fn_1431923

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
GPU2*0J 8� *\
fWRU
S__inference_transformer_encoder_16_layer_call_and_return_conditional_losses_1430398s
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
StatefulPartitionedCallStatefulPartitionedCall:'#
!
_user_specified_name	1431917:'#
!
_user_specified_name	1431915:'#
!
_user_specified_name	1431913:'#
!
_user_specified_name	1431911:'#
!
_user_specified_name	1431909:'#
!
_user_specified_name	1431907:'
#
!
_user_specified_name	1431905:'	#
!
_user_specified_name	1431903:'#
!
_user_specified_name	1431901:'#
!
_user_specified_name	1431899:'#
!
_user_specified_name	1431897:'#
!
_user_specified_name	1431895:'#
!
_user_specified_name	1431893:'#
!
_user_specified_name	1431891:'#
!
_user_specified_name	1431889:'#
!
_user_specified_name	1431887:S O
+
_output_shapes
:���������P	
 
_user_specified_nameinputs
��
�
S__inference_transformer_encoder_17_layer_call_and_return_conditional_losses_1430606

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
dropout_17/IdentityIdentity-Encoder-FeedForwardLayer_2_3/BiasAdd:output:0*
T0*+
_output_shapes
:���������P	�
Encoder-2nd-AdditionLayer-3/addAddV2#Encoder-1st-AdditionLayer-3/add:z:0dropout_17/Identity:output:0*
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
�
�
8__inference_transformer_encoder_16_layer_call_fn_1431884

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
GPU2*0J 8� *\
fWRU
S__inference_transformer_encoder_16_layer_call_and_return_conditional_losses_1429592s
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
StatefulPartitionedCallStatefulPartitionedCall:'#
!
_user_specified_name	1431878:'#
!
_user_specified_name	1431876:'#
!
_user_specified_name	1431874:'#
!
_user_specified_name	1431872:'#
!
_user_specified_name	1431870:'#
!
_user_specified_name	1431868:'
#
!
_user_specified_name	1431866:'	#
!
_user_specified_name	1431864:'#
!
_user_specified_name	1431862:'#
!
_user_specified_name	1431860:'#
!
_user_specified_name	1431858:'#
!
_user_specified_name	1431856:'#
!
_user_specified_name	1431854:'#
!
_user_specified_name	1431852:'#
!
_user_specified_name	1431850:'#
!
_user_specified_name	1431848:S O
+
_output_shapes
:���������P	
 
_user_specified_nameinputs
��
�
S__inference_transformer_encoder_15_layer_call_and_return_conditional_losses_1430190

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
dropout_15/IdentityIdentity-Encoder-FeedForwardLayer_2_1/BiasAdd:output:0*
T0*+
_output_shapes
:���������P	�
Encoder-2nd-AdditionLayer-1/addAddV2#Encoder-1st-AdditionLayer-1/add:z:0dropout_15/Identity:output:0*
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
� 
�
K__inference_FinalLayerNorm_layer_call_and_return_conditional_losses_1429890

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
w
M__inference_ConcatenateLayer_layer_call_and_return_conditional_losses_1429921

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
�
�
>__inference_FullyConnectedLayerAreaRatio_layer_call_fn_1432850

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
GPU2*0J 8� *b
f]R[
Y__inference_FullyConnectedLayerAreaRatio_layer_call_and_return_conditional_losses_1429964o
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
StatefulPartitionedCallStatefulPartitionedCall:'#
!
_user_specified_name	1432846:'#
!
_user_specified_name	1432844:O K
'
_output_shapes
:���������

 
_user_specified_nameinputs
�
e
I__inference_MaskingLayer_layer_call_and_return_conditional_losses_1431405

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
�
_
C__inference_Output_layer_call_and_return_conditional_losses_1430691

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
�
m
Q__inference_StandardizeTimeLimit_layer_call_and_return_conditional_losses_1430663

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
S__inference_transformer_encoder_17_layer_call_and_return_conditional_losses_1432551

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
:���������P	]
dropout_17/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?�
dropout_17/dropout/MulMul-Encoder-FeedForwardLayer_2_3/BiasAdd:output:0!dropout_17/dropout/Const:output:0*
T0*+
_output_shapes
:���������P	�
dropout_17/dropout/ShapeShape-Encoder-FeedForwardLayer_2_3/BiasAdd:output:0*
T0*
_output_shapes
::���
/dropout_17/dropout/random_uniform/RandomUniformRandomUniform!dropout_17/dropout/Shape:output:0*
T0*+
_output_shapes
:���������P	*
dtype0*
seed2*
seed��f
!dropout_17/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=�
dropout_17/dropout/GreaterEqualGreaterEqual8dropout_17/dropout/random_uniform/RandomUniform:output:0*dropout_17/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:���������P	_
dropout_17/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout_17/dropout/SelectV2SelectV2#dropout_17/dropout/GreaterEqual:z:0dropout_17/dropout/Mul:z:0#dropout_17/dropout/Const_1:output:0*
T0*+
_output_shapes
:���������P	�
Encoder-2nd-AdditionLayer-3/addAddV2#Encoder-1st-AdditionLayer-3/add:z:0$dropout_17/dropout/SelectV2:output:0*
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
y
]__inference_ReduceStackDimensionViaSummation_layer_call_and_return_conditional_losses_1429903

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
R
6__inference_StandardizeTimeLimit_layer_call_fn_1432807

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
GPU2*0J 8� *Z
fURS
Q__inference_StandardizeTimeLimit_layer_call_and_return_conditional_losses_1429913`
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
�
�
V__inference_FullyConnectedLayerSolved_layer_call_and_return_conditional_losses_1432895

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
�
^
B__inference_ReduceStackDimensionViaSummation_layer_call_fn_1432781

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
GPU2*0J 8� *f
faR_
]__inference_ReduceStackDimensionViaSummation_layer_call_and_return_conditional_losses_1429903`
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
��
�
S__inference_transformer_encoder_17_layer_call_and_return_conditional_losses_1432725

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
dropout_17/IdentityIdentity-Encoder-FeedForwardLayer_2_3/BiasAdd:output:0*
T0*+
_output_shapes
:���������P	�
Encoder-2nd-AdditionLayer-3/addAddV2#Encoder-1st-AdditionLayer-3/add:z:0dropout_17/Identity:output:0*
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
�{
�
D__inference_model_5_layer_call_and_return_conditional_losses_1430011
stacklevelinputfeatures
timelimitinput,
transformer_encoder_15_1429371:	,
transformer_encoder_15_1429373:	4
transformer_encoder_15_1429375:	0
transformer_encoder_15_1429377:4
transformer_encoder_15_1429379:	0
transformer_encoder_15_1429381:4
transformer_encoder_15_1429383:	0
transformer_encoder_15_1429385:4
transformer_encoder_15_1429387:	,
transformer_encoder_15_1429389:	,
transformer_encoder_15_1429391:	,
transformer_encoder_15_1429393:	0
transformer_encoder_15_1429395:		,
transformer_encoder_15_1429397:	0
transformer_encoder_15_1429399:		,
transformer_encoder_15_1429401:	,
transformer_encoder_16_1429593:	,
transformer_encoder_16_1429595:	4
transformer_encoder_16_1429597:	0
transformer_encoder_16_1429599:4
transformer_encoder_16_1429601:	0
transformer_encoder_16_1429603:4
transformer_encoder_16_1429605:	0
transformer_encoder_16_1429607:4
transformer_encoder_16_1429609:	,
transformer_encoder_16_1429611:	,
transformer_encoder_16_1429613:	,
transformer_encoder_16_1429615:	0
transformer_encoder_16_1429617:		,
transformer_encoder_16_1429619:	0
transformer_encoder_16_1429621:		,
transformer_encoder_16_1429623:	,
transformer_encoder_17_1429815:	,
transformer_encoder_17_1429817:	4
transformer_encoder_17_1429819:	0
transformer_encoder_17_1429821:4
transformer_encoder_17_1429823:	0
transformer_encoder_17_1429825:4
transformer_encoder_17_1429827:	0
transformer_encoder_17_1429829:4
transformer_encoder_17_1429831:	,
transformer_encoder_17_1429833:	,
transformer_encoder_17_1429835:	,
transformer_encoder_17_1429837:	0
transformer_encoder_17_1429839:		,
transformer_encoder_17_1429841:	0
transformer_encoder_17_1429843:		,
transformer_encoder_17_1429845:	$
finallayernorm_1429891:	$
finallayernorm_1429893:	3
!fullyconnectedlayersolved_1429942:

/
!fullyconnectedlayersolved_1429944:
6
$fullyconnectedlayerarearatio_1429965:

2
$fullyconnectedlayerarearatio_1429967:
*
predictionsolved_1429981:
&
predictionsolved_1429983:-
predictionarearatio_1429997:
)
predictionarearatio_1429999:
identity

identity_1��&FinalLayerNorm/StatefulPartitionedCall�4FullyConnectedLayerAreaRatio/StatefulPartitionedCall�1FullyConnectedLayerSolved/StatefulPartitionedCall�+PredictionAreaRatio/StatefulPartitionedCall�(PredictionSolved/StatefulPartitionedCall�.transformer_encoder_15/StatefulPartitionedCall�.transformer_encoder_16/StatefulPartitionedCall�.transformer_encoder_17/StatefulPartitionedCall�
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
GPU2*0J 8� *R
fMRK
I__inference_MaskingLayer_layer_call_and_return_conditional_losses_1429180�
transformer_encoder_15/CastCast%MaskingLayer/PartitionedCall:output:0*

DstT0*

SrcT0*+
_output_shapes
:���������P	�
.transformer_encoder_15/StatefulPartitionedCallStatefulPartitionedCalltransformer_encoder_15/Cast:y:0transformer_encoder_15_1429371transformer_encoder_15_1429373transformer_encoder_15_1429375transformer_encoder_15_1429377transformer_encoder_15_1429379transformer_encoder_15_1429381transformer_encoder_15_1429383transformer_encoder_15_1429385transformer_encoder_15_1429387transformer_encoder_15_1429389transformer_encoder_15_1429391transformer_encoder_15_1429393transformer_encoder_15_1429395transformer_encoder_15_1429397transformer_encoder_15_1429399transformer_encoder_15_1429401*
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
GPU2*0J 8� *\
fWRU
S__inference_transformer_encoder_15_layer_call_and_return_conditional_losses_1429370�
.transformer_encoder_16/StatefulPartitionedCallStatefulPartitionedCall7transformer_encoder_15/StatefulPartitionedCall:output:0transformer_encoder_16_1429593transformer_encoder_16_1429595transformer_encoder_16_1429597transformer_encoder_16_1429599transformer_encoder_16_1429601transformer_encoder_16_1429603transformer_encoder_16_1429605transformer_encoder_16_1429607transformer_encoder_16_1429609transformer_encoder_16_1429611transformer_encoder_16_1429613transformer_encoder_16_1429615transformer_encoder_16_1429617transformer_encoder_16_1429619transformer_encoder_16_1429621transformer_encoder_16_1429623*
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
GPU2*0J 8� *\
fWRU
S__inference_transformer_encoder_16_layer_call_and_return_conditional_losses_1429592�
.transformer_encoder_17/StatefulPartitionedCallStatefulPartitionedCall7transformer_encoder_16/StatefulPartitionedCall:output:0transformer_encoder_17_1429815transformer_encoder_17_1429817transformer_encoder_17_1429819transformer_encoder_17_1429821transformer_encoder_17_1429823transformer_encoder_17_1429825transformer_encoder_17_1429827transformer_encoder_17_1429829transformer_encoder_17_1429831transformer_encoder_17_1429833transformer_encoder_17_1429835transformer_encoder_17_1429837transformer_encoder_17_1429839transformer_encoder_17_1429841transformer_encoder_17_1429843transformer_encoder_17_1429845*
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
GPU2*0J 8� *\
fWRU
S__inference_transformer_encoder_17_layer_call_and_return_conditional_losses_1429814�
&FinalLayerNorm/StatefulPartitionedCallStatefulPartitionedCall7transformer_encoder_17/StatefulPartitionedCall:output:0finallayernorm_1429891finallayernorm_1429893*
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
GPU2*0J 8� *T
fORM
K__inference_FinalLayerNorm_layer_call_and_return_conditional_losses_1429890�
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
GPU2*0J 8� *f
faR_
]__inference_ReduceStackDimensionViaSummation_layer_call_and_return_conditional_losses_1429903r
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
GPU2*0J 8� *Z
fURS
Q__inference_StandardizeTimeLimit_layer_call_and_return_conditional_losses_1429913�
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
GPU2*0J 8� *V
fQRO
M__inference_ConcatenateLayer_layer_call_and_return_conditional_losses_1429921�
"ConcatenateLayer/PartitionedCall_1PartitionedCall9ReduceStackDimensionViaSummation/PartitionedCall:output:0-StandardizeTimeLimit/PartitionedCall:output:0*
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
GPU2*0J 8� *V
fQRO
M__inference_ConcatenateLayer_layer_call_and_return_conditional_losses_1429921�
1FullyConnectedLayerSolved/StatefulPartitionedCallStatefulPartitionedCall)ConcatenateLayer/PartitionedCall:output:0!fullyconnectedlayersolved_1429942!fullyconnectedlayersolved_1429944*
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
GPU2*0J 8� *_
fZRX
V__inference_FullyConnectedLayerSolved_layer_call_and_return_conditional_losses_1429941�
4FullyConnectedLayerAreaRatio/StatefulPartitionedCallStatefulPartitionedCall+ConcatenateLayer/PartitionedCall_1:output:0$fullyconnectedlayerarearatio_1429965$fullyconnectedlayerarearatio_1429967*
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
GPU2*0J 8� *b
f]R[
Y__inference_FullyConnectedLayerAreaRatio_layer_call_and_return_conditional_losses_1429964�
(PredictionSolved/StatefulPartitionedCallStatefulPartitionedCall:FullyConnectedLayerSolved/StatefulPartitionedCall:output:0predictionsolved_1429981predictionsolved_1429983*
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
GPU2*0J 8� *V
fQRO
M__inference_PredictionSolved_layer_call_and_return_conditional_losses_1429980�
+PredictionAreaRatio/StatefulPartitionedCallStatefulPartitionedCall=FullyConnectedLayerAreaRatio/StatefulPartitionedCall:output:0predictionarearatio_1429997predictionarearatio_1429999*
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
GPU2*0J 8� *Y
fTRR
P__inference_PredictionAreaRatio_layer_call_and_return_conditional_losses_1429996�
Output/PartitionedCallPartitionedCall1PredictionSolved/StatefulPartitionedCall:output:0*
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
GPU2*0J 8� *L
fGRE
C__inference_Output_layer_call_and_return_conditional_losses_1430006�
Output/PartitionedCall_1PartitionedCall4PredictionAreaRatio/StatefulPartitionedCall:output:0*
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
GPU2*0J 8� *L
fGRE
C__inference_Output_layer_call_and_return_conditional_losses_1430006a
IdentityIdentity!Output/PartitionedCall_1:output:0^NoOp*
T0*
_output_shapes
:a

Identity_1IdentityOutput/PartitionedCall:output:0^NoOp*
T0*
_output_shapes
:�
NoOpNoOp'^FinalLayerNorm/StatefulPartitionedCall5^FullyConnectedLayerAreaRatio/StatefulPartitionedCall2^FullyConnectedLayerSolved/StatefulPartitionedCall,^PredictionAreaRatio/StatefulPartitionedCall)^PredictionSolved/StatefulPartitionedCall/^transformer_encoder_15/StatefulPartitionedCall/^transformer_encoder_16/StatefulPartitionedCall/^transformer_encoder_17/StatefulPartitionedCall*
_output_shapes
 "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:���������P	:���������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2P
&FinalLayerNorm/StatefulPartitionedCall&FinalLayerNorm/StatefulPartitionedCall2l
4FullyConnectedLayerAreaRatio/StatefulPartitionedCall4FullyConnectedLayerAreaRatio/StatefulPartitionedCall2f
1FullyConnectedLayerSolved/StatefulPartitionedCall1FullyConnectedLayerSolved/StatefulPartitionedCall2Z
+PredictionAreaRatio/StatefulPartitionedCall+PredictionAreaRatio/StatefulPartitionedCall2T
(PredictionSolved/StatefulPartitionedCall(PredictionSolved/StatefulPartitionedCall2`
.transformer_encoder_15/StatefulPartitionedCall.transformer_encoder_15/StatefulPartitionedCall2`
.transformer_encoder_16/StatefulPartitionedCall.transformer_encoder_16/StatefulPartitionedCall2`
.transformer_encoder_17/StatefulPartitionedCall.transformer_encoder_17/StatefulPartitionedCall:';#
!
_user_specified_name	1429999:':#
!
_user_specified_name	1429997:'9#
!
_user_specified_name	1429983:'8#
!
_user_specified_name	1429981:'7#
!
_user_specified_name	1429967:'6#
!
_user_specified_name	1429965:'5#
!
_user_specified_name	1429944:'4#
!
_user_specified_name	1429942:'3#
!
_user_specified_name	1429893:'2#
!
_user_specified_name	1429891:'1#
!
_user_specified_name	1429845:'0#
!
_user_specified_name	1429843:'/#
!
_user_specified_name	1429841:'.#
!
_user_specified_name	1429839:'-#
!
_user_specified_name	1429837:',#
!
_user_specified_name	1429835:'+#
!
_user_specified_name	1429833:'*#
!
_user_specified_name	1429831:')#
!
_user_specified_name	1429829:'(#
!
_user_specified_name	1429827:''#
!
_user_specified_name	1429825:'&#
!
_user_specified_name	1429823:'%#
!
_user_specified_name	1429821:'$#
!
_user_specified_name	1429819:'##
!
_user_specified_name	1429817:'"#
!
_user_specified_name	1429815:'!#
!
_user_specified_name	1429623:' #
!
_user_specified_name	1429621:'#
!
_user_specified_name	1429619:'#
!
_user_specified_name	1429617:'#
!
_user_specified_name	1429615:'#
!
_user_specified_name	1429613:'#
!
_user_specified_name	1429611:'#
!
_user_specified_name	1429609:'#
!
_user_specified_name	1429607:'#
!
_user_specified_name	1429605:'#
!
_user_specified_name	1429603:'#
!
_user_specified_name	1429601:'#
!
_user_specified_name	1429599:'#
!
_user_specified_name	1429597:'#
!
_user_specified_name	1429595:'#
!
_user_specified_name	1429593:'#
!
_user_specified_name	1429401:'#
!
_user_specified_name	1429399:'#
!
_user_specified_name	1429397:'#
!
_user_specified_name	1429395:'#
!
_user_specified_name	1429393:'#
!
_user_specified_name	1429391:'#
!
_user_specified_name	1429389:'
#
!
_user_specified_name	1429387:'	#
!
_user_specified_name	1429385:'#
!
_user_specified_name	1429383:'#
!
_user_specified_name	1429381:'#
!
_user_specified_name	1429379:'#
!
_user_specified_name	1429377:'#
!
_user_specified_name	1429375:'#
!
_user_specified_name	1429373:'#
!
_user_specified_name	1429371:WS
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
��	
�W
"__inference__wrapped_model_1429166
stacklevelinputfeatures
timelimitinputi
[model_5_transformer_encoder_15_encoder_1st_normalizationlayer_1_mul_readvariableop_resource:	i
[model_5_transformer_encoder_15_encoder_1st_normalizationlayer_1_add_readvariableop_resource:	}
gmodel_5_transformer_encoder_15_encoder_selfattentionlayer_1_query_einsum_einsum_readvariableop_resource:	o
]model_5_transformer_encoder_15_encoder_selfattentionlayer_1_query_add_readvariableop_resource:{
emodel_5_transformer_encoder_15_encoder_selfattentionlayer_1_key_einsum_einsum_readvariableop_resource:	m
[model_5_transformer_encoder_15_encoder_selfattentionlayer_1_key_add_readvariableop_resource:}
gmodel_5_transformer_encoder_15_encoder_selfattentionlayer_1_value_einsum_einsum_readvariableop_resource:	o
]model_5_transformer_encoder_15_encoder_selfattentionlayer_1_value_add_readvariableop_resource:�
rmodel_5_transformer_encoder_15_encoder_selfattentionlayer_1_attention_output_einsum_einsum_readvariableop_resource:	v
hmodel_5_transformer_encoder_15_encoder_selfattentionlayer_1_attention_output_add_readvariableop_resource:	i
[model_5_transformer_encoder_15_encoder_2nd_normalizationlayer_1_mul_readvariableop_resource:	i
[model_5_transformer_encoder_15_encoder_2nd_normalizationlayer_1_add_readvariableop_resource:	o
]model_5_transformer_encoder_15_encoder_feedforwardlayer_1_1_tensordot_readvariableop_resource:		i
[model_5_transformer_encoder_15_encoder_feedforwardlayer_1_1_biasadd_readvariableop_resource:	o
]model_5_transformer_encoder_15_encoder_feedforwardlayer_2_1_tensordot_readvariableop_resource:		i
[model_5_transformer_encoder_15_encoder_feedforwardlayer_2_1_biasadd_readvariableop_resource:	i
[model_5_transformer_encoder_16_encoder_1st_normalizationlayer_2_mul_readvariableop_resource:	i
[model_5_transformer_encoder_16_encoder_1st_normalizationlayer_2_add_readvariableop_resource:	}
gmodel_5_transformer_encoder_16_encoder_selfattentionlayer_2_query_einsum_einsum_readvariableop_resource:	o
]model_5_transformer_encoder_16_encoder_selfattentionlayer_2_query_add_readvariableop_resource:{
emodel_5_transformer_encoder_16_encoder_selfattentionlayer_2_key_einsum_einsum_readvariableop_resource:	m
[model_5_transformer_encoder_16_encoder_selfattentionlayer_2_key_add_readvariableop_resource:}
gmodel_5_transformer_encoder_16_encoder_selfattentionlayer_2_value_einsum_einsum_readvariableop_resource:	o
]model_5_transformer_encoder_16_encoder_selfattentionlayer_2_value_add_readvariableop_resource:�
rmodel_5_transformer_encoder_16_encoder_selfattentionlayer_2_attention_output_einsum_einsum_readvariableop_resource:	v
hmodel_5_transformer_encoder_16_encoder_selfattentionlayer_2_attention_output_add_readvariableop_resource:	i
[model_5_transformer_encoder_16_encoder_2nd_normalizationlayer_2_mul_readvariableop_resource:	i
[model_5_transformer_encoder_16_encoder_2nd_normalizationlayer_2_add_readvariableop_resource:	o
]model_5_transformer_encoder_16_encoder_feedforwardlayer_1_2_tensordot_readvariableop_resource:		i
[model_5_transformer_encoder_16_encoder_feedforwardlayer_1_2_biasadd_readvariableop_resource:	o
]model_5_transformer_encoder_16_encoder_feedforwardlayer_2_2_tensordot_readvariableop_resource:		i
[model_5_transformer_encoder_16_encoder_feedforwardlayer_2_2_biasadd_readvariableop_resource:	i
[model_5_transformer_encoder_17_encoder_1st_normalizationlayer_3_mul_readvariableop_resource:	i
[model_5_transformer_encoder_17_encoder_1st_normalizationlayer_3_add_readvariableop_resource:	}
gmodel_5_transformer_encoder_17_encoder_selfattentionlayer_3_query_einsum_einsum_readvariableop_resource:	o
]model_5_transformer_encoder_17_encoder_selfattentionlayer_3_query_add_readvariableop_resource:{
emodel_5_transformer_encoder_17_encoder_selfattentionlayer_3_key_einsum_einsum_readvariableop_resource:	m
[model_5_transformer_encoder_17_encoder_selfattentionlayer_3_key_add_readvariableop_resource:}
gmodel_5_transformer_encoder_17_encoder_selfattentionlayer_3_value_einsum_einsum_readvariableop_resource:	o
]model_5_transformer_encoder_17_encoder_selfattentionlayer_3_value_add_readvariableop_resource:�
rmodel_5_transformer_encoder_17_encoder_selfattentionlayer_3_attention_output_einsum_einsum_readvariableop_resource:	v
hmodel_5_transformer_encoder_17_encoder_selfattentionlayer_3_attention_output_add_readvariableop_resource:	i
[model_5_transformer_encoder_17_encoder_2nd_normalizationlayer_3_mul_readvariableop_resource:	i
[model_5_transformer_encoder_17_encoder_2nd_normalizationlayer_3_add_readvariableop_resource:	o
]model_5_transformer_encoder_17_encoder_feedforwardlayer_1_3_tensordot_readvariableop_resource:		i
[model_5_transformer_encoder_17_encoder_feedforwardlayer_1_3_biasadd_readvariableop_resource:	o
]model_5_transformer_encoder_17_encoder_feedforwardlayer_2_3_tensordot_readvariableop_resource:		i
[model_5_transformer_encoder_17_encoder_feedforwardlayer_2_3_biasadd_readvariableop_resource:	@
2model_5_finallayernorm_mul_readvariableop_resource:	@
2model_5_finallayernorm_add_readvariableop_resource:	R
@model_5_fullyconnectedlayersolved_matmul_readvariableop_resource:

O
Amodel_5_fullyconnectedlayersolved_biasadd_readvariableop_resource:
U
Cmodel_5_fullyconnectedlayerarearatio_matmul_readvariableop_resource:

R
Dmodel_5_fullyconnectedlayerarearatio_biasadd_readvariableop_resource:
I
7model_5_predictionsolved_matmul_readvariableop_resource:
F
8model_5_predictionsolved_biasadd_readvariableop_resource:L
:model_5_predictionarearatio_matmul_readvariableop_resource:
I
;model_5_predictionarearatio_biasadd_readvariableop_resource:
identity

identity_1��)model_5/FinalLayerNorm/add/ReadVariableOp�)model_5/FinalLayerNorm/mul/ReadVariableOp�;model_5/FullyConnectedLayerAreaRatio/BiasAdd/ReadVariableOp�:model_5/FullyConnectedLayerAreaRatio/MatMul/ReadVariableOp�8model_5/FullyConnectedLayerSolved/BiasAdd/ReadVariableOp�7model_5/FullyConnectedLayerSolved/MatMul/ReadVariableOp�2model_5/PredictionAreaRatio/BiasAdd/ReadVariableOp�1model_5/PredictionAreaRatio/MatMul/ReadVariableOp�/model_5/PredictionSolved/BiasAdd/ReadVariableOp�.model_5/PredictionSolved/MatMul/ReadVariableOp�Rmodel_5/transformer_encoder_15/Encoder-1st-NormalizationLayer-1/add/ReadVariableOp�Rmodel_5/transformer_encoder_15/Encoder-1st-NormalizationLayer-1/mul/ReadVariableOp�Rmodel_5/transformer_encoder_15/Encoder-2nd-NormalizationLayer-1/add/ReadVariableOp�Rmodel_5/transformer_encoder_15/Encoder-2nd-NormalizationLayer-1/mul/ReadVariableOp�Rmodel_5/transformer_encoder_15/Encoder-FeedForwardLayer_1_1/BiasAdd/ReadVariableOp�Tmodel_5/transformer_encoder_15/Encoder-FeedForwardLayer_1_1/Tensordot/ReadVariableOp�Rmodel_5/transformer_encoder_15/Encoder-FeedForwardLayer_2_1/BiasAdd/ReadVariableOp�Tmodel_5/transformer_encoder_15/Encoder-FeedForwardLayer_2_1/Tensordot/ReadVariableOp�_model_5/transformer_encoder_15/Encoder-SelfAttentionLayer-1/attention_output/add/ReadVariableOp�imodel_5/transformer_encoder_15/Encoder-SelfAttentionLayer-1/attention_output/einsum/Einsum/ReadVariableOp�Rmodel_5/transformer_encoder_15/Encoder-SelfAttentionLayer-1/key/add/ReadVariableOp�\model_5/transformer_encoder_15/Encoder-SelfAttentionLayer-1/key/einsum/Einsum/ReadVariableOp�Tmodel_5/transformer_encoder_15/Encoder-SelfAttentionLayer-1/query/add/ReadVariableOp�^model_5/transformer_encoder_15/Encoder-SelfAttentionLayer-1/query/einsum/Einsum/ReadVariableOp�Tmodel_5/transformer_encoder_15/Encoder-SelfAttentionLayer-1/value/add/ReadVariableOp�^model_5/transformer_encoder_15/Encoder-SelfAttentionLayer-1/value/einsum/Einsum/ReadVariableOp�Rmodel_5/transformer_encoder_16/Encoder-1st-NormalizationLayer-2/add/ReadVariableOp�Rmodel_5/transformer_encoder_16/Encoder-1st-NormalizationLayer-2/mul/ReadVariableOp�Rmodel_5/transformer_encoder_16/Encoder-2nd-NormalizationLayer-2/add/ReadVariableOp�Rmodel_5/transformer_encoder_16/Encoder-2nd-NormalizationLayer-2/mul/ReadVariableOp�Rmodel_5/transformer_encoder_16/Encoder-FeedForwardLayer_1_2/BiasAdd/ReadVariableOp�Tmodel_5/transformer_encoder_16/Encoder-FeedForwardLayer_1_2/Tensordot/ReadVariableOp�Rmodel_5/transformer_encoder_16/Encoder-FeedForwardLayer_2_2/BiasAdd/ReadVariableOp�Tmodel_5/transformer_encoder_16/Encoder-FeedForwardLayer_2_2/Tensordot/ReadVariableOp�_model_5/transformer_encoder_16/Encoder-SelfAttentionLayer-2/attention_output/add/ReadVariableOp�imodel_5/transformer_encoder_16/Encoder-SelfAttentionLayer-2/attention_output/einsum/Einsum/ReadVariableOp�Rmodel_5/transformer_encoder_16/Encoder-SelfAttentionLayer-2/key/add/ReadVariableOp�\model_5/transformer_encoder_16/Encoder-SelfAttentionLayer-2/key/einsum/Einsum/ReadVariableOp�Tmodel_5/transformer_encoder_16/Encoder-SelfAttentionLayer-2/query/add/ReadVariableOp�^model_5/transformer_encoder_16/Encoder-SelfAttentionLayer-2/query/einsum/Einsum/ReadVariableOp�Tmodel_5/transformer_encoder_16/Encoder-SelfAttentionLayer-2/value/add/ReadVariableOp�^model_5/transformer_encoder_16/Encoder-SelfAttentionLayer-2/value/einsum/Einsum/ReadVariableOp�Rmodel_5/transformer_encoder_17/Encoder-1st-NormalizationLayer-3/add/ReadVariableOp�Rmodel_5/transformer_encoder_17/Encoder-1st-NormalizationLayer-3/mul/ReadVariableOp�Rmodel_5/transformer_encoder_17/Encoder-2nd-NormalizationLayer-3/add/ReadVariableOp�Rmodel_5/transformer_encoder_17/Encoder-2nd-NormalizationLayer-3/mul/ReadVariableOp�Rmodel_5/transformer_encoder_17/Encoder-FeedForwardLayer_1_3/BiasAdd/ReadVariableOp�Tmodel_5/transformer_encoder_17/Encoder-FeedForwardLayer_1_3/Tensordot/ReadVariableOp�Rmodel_5/transformer_encoder_17/Encoder-FeedForwardLayer_2_3/BiasAdd/ReadVariableOp�Tmodel_5/transformer_encoder_17/Encoder-FeedForwardLayer_2_3/Tensordot/ReadVariableOp�_model_5/transformer_encoder_17/Encoder-SelfAttentionLayer-3/attention_output/add/ReadVariableOp�imodel_5/transformer_encoder_17/Encoder-SelfAttentionLayer-3/attention_output/einsum/Einsum/ReadVariableOp�Rmodel_5/transformer_encoder_17/Encoder-SelfAttentionLayer-3/key/add/ReadVariableOp�\model_5/transformer_encoder_17/Encoder-SelfAttentionLayer-3/key/einsum/Einsum/ReadVariableOp�Tmodel_5/transformer_encoder_17/Encoder-SelfAttentionLayer-3/query/add/ReadVariableOp�^model_5/transformer_encoder_17/Encoder-SelfAttentionLayer-3/query/einsum/Einsum/ReadVariableOp�Tmodel_5/transformer_encoder_17/Encoder-SelfAttentionLayer-3/value/add/ReadVariableOp�^model_5/transformer_encoder_17/Encoder-SelfAttentionLayer-3/value/einsum/Einsum/ReadVariableOpa
model_5/MaskingLayer/NotEqual/yConst*
_output_shapes
: *
dtype0*
value	B j �
model_5/MaskingLayer/NotEqualNotEqualstacklevelinputfeatures(model_5/MaskingLayer/NotEqual/y:output:0*
T0*+
_output_shapes
:���������P	u
*model_5/MaskingLayer/Any/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
����������
model_5/MaskingLayer/AnyAny!model_5/MaskingLayer/NotEqual:z:03model_5/MaskingLayer/Any/reduction_indices:output:0*+
_output_shapes
:���������P*
	keep_dims(�
model_5/MaskingLayer/CastCast!model_5/MaskingLayer/Any:output:0*

DstT0*

SrcT0
*+
_output_shapes
:���������P�
model_5/MaskingLayer/mulMulstacklevelinputfeaturesmodel_5/MaskingLayer/Cast:y:0*
T0*+
_output_shapes
:���������P	�
model_5/MaskingLayer/SqueezeSqueeze!model_5/MaskingLayer/Any:output:0*
T0
*'
_output_shapes
:���������P*
squeeze_dims

����������
#model_5/transformer_encoder_15/CastCastmodel_5/MaskingLayer/mul:z:0*

DstT0*

SrcT0*+
_output_shapes
:���������P	�
Emodel_5/transformer_encoder_15/Encoder-1st-NormalizationLayer-1/ShapeShape'model_5/transformer_encoder_15/Cast:y:0*
T0*
_output_shapes
::���
Smodel_5/transformer_encoder_15/Encoder-1st-NormalizationLayer-1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
Umodel_5/transformer_encoder_15/Encoder-1st-NormalizationLayer-1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
Umodel_5/transformer_encoder_15/Encoder-1st-NormalizationLayer-1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
Mmodel_5/transformer_encoder_15/Encoder-1st-NormalizationLayer-1/strided_sliceStridedSliceNmodel_5/transformer_encoder_15/Encoder-1st-NormalizationLayer-1/Shape:output:0\model_5/transformer_encoder_15/Encoder-1st-NormalizationLayer-1/strided_slice/stack:output:0^model_5/transformer_encoder_15/Encoder-1st-NormalizationLayer-1/strided_slice/stack_1:output:0^model_5/transformer_encoder_15/Encoder-1st-NormalizationLayer-1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask�
Emodel_5/transformer_encoder_15/Encoder-1st-NormalizationLayer-1/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
Dmodel_5/transformer_encoder_15/Encoder-1st-NormalizationLayer-1/ProdProdVmodel_5/transformer_encoder_15/Encoder-1st-NormalizationLayer-1/strided_slice:output:0Nmodel_5/transformer_encoder_15/Encoder-1st-NormalizationLayer-1/Const:output:0*
T0*
_output_shapes
: �
Umodel_5/transformer_encoder_15/Encoder-1st-NormalizationLayer-1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:�
Wmodel_5/transformer_encoder_15/Encoder-1st-NormalizationLayer-1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: �
Wmodel_5/transformer_encoder_15/Encoder-1st-NormalizationLayer-1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
Omodel_5/transformer_encoder_15/Encoder-1st-NormalizationLayer-1/strided_slice_1StridedSliceNmodel_5/transformer_encoder_15/Encoder-1st-NormalizationLayer-1/Shape:output:0^model_5/transformer_encoder_15/Encoder-1st-NormalizationLayer-1/strided_slice_1/stack:output:0`model_5/transformer_encoder_15/Encoder-1st-NormalizationLayer-1/strided_slice_1/stack_1:output:0`model_5/transformer_encoder_15/Encoder-1st-NormalizationLayer-1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask�
Gmodel_5/transformer_encoder_15/Encoder-1st-NormalizationLayer-1/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
Fmodel_5/transformer_encoder_15/Encoder-1st-NormalizationLayer-1/Prod_1ProdXmodel_5/transformer_encoder_15/Encoder-1st-NormalizationLayer-1/strided_slice_1:output:0Pmodel_5/transformer_encoder_15/Encoder-1st-NormalizationLayer-1/Const_1:output:0*
T0*
_output_shapes
: �
Omodel_5/transformer_encoder_15/Encoder-1st-NormalizationLayer-1/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :�
Omodel_5/transformer_encoder_15/Encoder-1st-NormalizationLayer-1/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :�
Mmodel_5/transformer_encoder_15/Encoder-1st-NormalizationLayer-1/Reshape/shapePackXmodel_5/transformer_encoder_15/Encoder-1st-NormalizationLayer-1/Reshape/shape/0:output:0Mmodel_5/transformer_encoder_15/Encoder-1st-NormalizationLayer-1/Prod:output:0Omodel_5/transformer_encoder_15/Encoder-1st-NormalizationLayer-1/Prod_1:output:0Xmodel_5/transformer_encoder_15/Encoder-1st-NormalizationLayer-1/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:�
Gmodel_5/transformer_encoder_15/Encoder-1st-NormalizationLayer-1/ReshapeReshape'model_5/transformer_encoder_15/Cast:y:0Vmodel_5/transformer_encoder_15/Encoder-1st-NormalizationLayer-1/Reshape/shape:output:0*
T0*8
_output_shapes&
$:"�������������������
Kmodel_5/transformer_encoder_15/Encoder-1st-NormalizationLayer-1/ones/packedPackMmodel_5/transformer_encoder_15/Encoder-1st-NormalizationLayer-1/Prod:output:0*
N*
T0*
_output_shapes
:�
Jmodel_5/transformer_encoder_15/Encoder-1st-NormalizationLayer-1/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
Dmodel_5/transformer_encoder_15/Encoder-1st-NormalizationLayer-1/onesFillTmodel_5/transformer_encoder_15/Encoder-1st-NormalizationLayer-1/ones/packed:output:0Smodel_5/transformer_encoder_15/Encoder-1st-NormalizationLayer-1/ones/Const:output:0*
T0*#
_output_shapes
:����������
Lmodel_5/transformer_encoder_15/Encoder-1st-NormalizationLayer-1/zeros/packedPackMmodel_5/transformer_encoder_15/Encoder-1st-NormalizationLayer-1/Prod:output:0*
N*
T0*
_output_shapes
:�
Kmodel_5/transformer_encoder_15/Encoder-1st-NormalizationLayer-1/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
Emodel_5/transformer_encoder_15/Encoder-1st-NormalizationLayer-1/zerosFillUmodel_5/transformer_encoder_15/Encoder-1st-NormalizationLayer-1/zeros/packed:output:0Tmodel_5/transformer_encoder_15/Encoder-1st-NormalizationLayer-1/zeros/Const:output:0*
T0*#
_output_shapes
:����������
Gmodel_5/transformer_encoder_15/Encoder-1st-NormalizationLayer-1/Const_2Const*
_output_shapes
: *
dtype0*
valueB �
Gmodel_5/transformer_encoder_15/Encoder-1st-NormalizationLayer-1/Const_3Const*
_output_shapes
: *
dtype0*
valueB �
Pmodel_5/transformer_encoder_15/Encoder-1st-NormalizationLayer-1/FusedBatchNormV3FusedBatchNormV3Pmodel_5/transformer_encoder_15/Encoder-1st-NormalizationLayer-1/Reshape:output:0Mmodel_5/transformer_encoder_15/Encoder-1st-NormalizationLayer-1/ones:output:0Nmodel_5/transformer_encoder_15/Encoder-1st-NormalizationLayer-1/zeros:output:0Pmodel_5/transformer_encoder_15/Encoder-1st-NormalizationLayer-1/Const_2:output:0Pmodel_5/transformer_encoder_15/Encoder-1st-NormalizationLayer-1/Const_3:output:0*
T0*
U0*x
_output_shapesf
d:"������������������:���������:���������:���������:���������:*
data_formatNCHW*
epsilon%o�:�
Imodel_5/transformer_encoder_15/Encoder-1st-NormalizationLayer-1/Reshape_1ReshapeTmodel_5/transformer_encoder_15/Encoder-1st-NormalizationLayer-1/FusedBatchNormV3:y:0Nmodel_5/transformer_encoder_15/Encoder-1st-NormalizationLayer-1/Shape:output:0*
T0*+
_output_shapes
:���������P	�
Rmodel_5/transformer_encoder_15/Encoder-1st-NormalizationLayer-1/mul/ReadVariableOpReadVariableOp[model_5_transformer_encoder_15_encoder_1st_normalizationlayer_1_mul_readvariableop_resource*
_output_shapes
:	*
dtype0�
Cmodel_5/transformer_encoder_15/Encoder-1st-NormalizationLayer-1/mulMulRmodel_5/transformer_encoder_15/Encoder-1st-NormalizationLayer-1/Reshape_1:output:0Zmodel_5/transformer_encoder_15/Encoder-1st-NormalizationLayer-1/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������P	�
Rmodel_5/transformer_encoder_15/Encoder-1st-NormalizationLayer-1/add/ReadVariableOpReadVariableOp[model_5_transformer_encoder_15_encoder_1st_normalizationlayer_1_add_readvariableop_resource*
_output_shapes
:	*
dtype0�
Cmodel_5/transformer_encoder_15/Encoder-1st-NormalizationLayer-1/addAddV2Gmodel_5/transformer_encoder_15/Encoder-1st-NormalizationLayer-1/mul:z:0Zmodel_5/transformer_encoder_15/Encoder-1st-NormalizationLayer-1/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������P	�
^model_5/transformer_encoder_15/Encoder-SelfAttentionLayer-1/query/einsum/Einsum/ReadVariableOpReadVariableOpgmodel_5_transformer_encoder_15_encoder_selfattentionlayer_1_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:	*
dtype0�
Omodel_5/transformer_encoder_15/Encoder-SelfAttentionLayer-1/query/einsum/EinsumEinsumGmodel_5/transformer_encoder_15/Encoder-1st-NormalizationLayer-1/add:z:0fmodel_5/transformer_encoder_15/Encoder-SelfAttentionLayer-1/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������P*
equationabc,cde->abde�
Tmodel_5/transformer_encoder_15/Encoder-SelfAttentionLayer-1/query/add/ReadVariableOpReadVariableOp]model_5_transformer_encoder_15_encoder_selfattentionlayer_1_query_add_readvariableop_resource*
_output_shapes

:*
dtype0�
Emodel_5/transformer_encoder_15/Encoder-SelfAttentionLayer-1/query/addAddV2Xmodel_5/transformer_encoder_15/Encoder-SelfAttentionLayer-1/query/einsum/Einsum:output:0\model_5/transformer_encoder_15/Encoder-SelfAttentionLayer-1/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������P�
\model_5/transformer_encoder_15/Encoder-SelfAttentionLayer-1/key/einsum/Einsum/ReadVariableOpReadVariableOpemodel_5_transformer_encoder_15_encoder_selfattentionlayer_1_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:	*
dtype0�
Mmodel_5/transformer_encoder_15/Encoder-SelfAttentionLayer-1/key/einsum/EinsumEinsumGmodel_5/transformer_encoder_15/Encoder-1st-NormalizationLayer-1/add:z:0dmodel_5/transformer_encoder_15/Encoder-SelfAttentionLayer-1/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������P*
equationabc,cde->abde�
Rmodel_5/transformer_encoder_15/Encoder-SelfAttentionLayer-1/key/add/ReadVariableOpReadVariableOp[model_5_transformer_encoder_15_encoder_selfattentionlayer_1_key_add_readvariableop_resource*
_output_shapes

:*
dtype0�
Cmodel_5/transformer_encoder_15/Encoder-SelfAttentionLayer-1/key/addAddV2Vmodel_5/transformer_encoder_15/Encoder-SelfAttentionLayer-1/key/einsum/Einsum:output:0Zmodel_5/transformer_encoder_15/Encoder-SelfAttentionLayer-1/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������P�
^model_5/transformer_encoder_15/Encoder-SelfAttentionLayer-1/value/einsum/Einsum/ReadVariableOpReadVariableOpgmodel_5_transformer_encoder_15_encoder_selfattentionlayer_1_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:	*
dtype0�
Omodel_5/transformer_encoder_15/Encoder-SelfAttentionLayer-1/value/einsum/EinsumEinsumGmodel_5/transformer_encoder_15/Encoder-1st-NormalizationLayer-1/add:z:0fmodel_5/transformer_encoder_15/Encoder-SelfAttentionLayer-1/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������P*
equationabc,cde->abde�
Tmodel_5/transformer_encoder_15/Encoder-SelfAttentionLayer-1/value/add/ReadVariableOpReadVariableOp]model_5_transformer_encoder_15_encoder_selfattentionlayer_1_value_add_readvariableop_resource*
_output_shapes

:*
dtype0�
Emodel_5/transformer_encoder_15/Encoder-SelfAttentionLayer-1/value/addAddV2Xmodel_5/transformer_encoder_15/Encoder-SelfAttentionLayer-1/value/einsum/Einsum:output:0\model_5/transformer_encoder_15/Encoder-SelfAttentionLayer-1/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������P�
Amodel_5/transformer_encoder_15/Encoder-SelfAttentionLayer-1/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *:�?�
?model_5/transformer_encoder_15/Encoder-SelfAttentionLayer-1/MulMulImodel_5/transformer_encoder_15/Encoder-SelfAttentionLayer-1/query/add:z:0Jmodel_5/transformer_encoder_15/Encoder-SelfAttentionLayer-1/Mul/y:output:0*
T0*/
_output_shapes
:���������P�
Imodel_5/transformer_encoder_15/Encoder-SelfAttentionLayer-1/einsum/EinsumEinsumGmodel_5/transformer_encoder_15/Encoder-SelfAttentionLayer-1/key/add:z:0Cmodel_5/transformer_encoder_15/Encoder-SelfAttentionLayer-1/Mul:z:0*
N*
T0*/
_output_shapes
:���������PP*
equationaecd,abcd->acbe�
Kmodel_5/transformer_encoder_15/Encoder-SelfAttentionLayer-1/softmax/SoftmaxSoftmaxRmodel_5/transformer_encoder_15/Encoder-SelfAttentionLayer-1/einsum/Einsum:output:0*
T0*/
_output_shapes
:���������PP�
Lmodel_5/transformer_encoder_15/Encoder-SelfAttentionLayer-1/dropout/IdentityIdentityUmodel_5/transformer_encoder_15/Encoder-SelfAttentionLayer-1/softmax/Softmax:softmax:0*
T0*/
_output_shapes
:���������PP�
Kmodel_5/transformer_encoder_15/Encoder-SelfAttentionLayer-1/einsum_1/EinsumEinsumUmodel_5/transformer_encoder_15/Encoder-SelfAttentionLayer-1/dropout/Identity:output:0Imodel_5/transformer_encoder_15/Encoder-SelfAttentionLayer-1/value/add:z:0*
N*
T0*/
_output_shapes
:���������P*
equationacbe,aecd->abcd�
imodel_5/transformer_encoder_15/Encoder-SelfAttentionLayer-1/attention_output/einsum/Einsum/ReadVariableOpReadVariableOprmodel_5_transformer_encoder_15_encoder_selfattentionlayer_1_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:	*
dtype0�
Zmodel_5/transformer_encoder_15/Encoder-SelfAttentionLayer-1/attention_output/einsum/EinsumEinsumTmodel_5/transformer_encoder_15/Encoder-SelfAttentionLayer-1/einsum_1/Einsum:output:0qmodel_5/transformer_encoder_15/Encoder-SelfAttentionLayer-1/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:���������P	*
equationabcd,cde->abe�
_model_5/transformer_encoder_15/Encoder-SelfAttentionLayer-1/attention_output/add/ReadVariableOpReadVariableOphmodel_5_transformer_encoder_15_encoder_selfattentionlayer_1_attention_output_add_readvariableop_resource*
_output_shapes
:	*
dtype0�
Pmodel_5/transformer_encoder_15/Encoder-SelfAttentionLayer-1/attention_output/addAddV2cmodel_5/transformer_encoder_15/Encoder-SelfAttentionLayer-1/attention_output/einsum/Einsum:output:0gmodel_5/transformer_encoder_15/Encoder-SelfAttentionLayer-1/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������P	�
>model_5/transformer_encoder_15/Encoder-1st-AdditionLayer-1/addAddV2'model_5/transformer_encoder_15/Cast:y:0Tmodel_5/transformer_encoder_15/Encoder-SelfAttentionLayer-1/attention_output/add:z:0*
T0*+
_output_shapes
:���������P	�
Emodel_5/transformer_encoder_15/Encoder-2nd-NormalizationLayer-1/ShapeShapeBmodel_5/transformer_encoder_15/Encoder-1st-AdditionLayer-1/add:z:0*
T0*
_output_shapes
::���
Smodel_5/transformer_encoder_15/Encoder-2nd-NormalizationLayer-1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
Umodel_5/transformer_encoder_15/Encoder-2nd-NormalizationLayer-1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
Umodel_5/transformer_encoder_15/Encoder-2nd-NormalizationLayer-1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
Mmodel_5/transformer_encoder_15/Encoder-2nd-NormalizationLayer-1/strided_sliceStridedSliceNmodel_5/transformer_encoder_15/Encoder-2nd-NormalizationLayer-1/Shape:output:0\model_5/transformer_encoder_15/Encoder-2nd-NormalizationLayer-1/strided_slice/stack:output:0^model_5/transformer_encoder_15/Encoder-2nd-NormalizationLayer-1/strided_slice/stack_1:output:0^model_5/transformer_encoder_15/Encoder-2nd-NormalizationLayer-1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask�
Emodel_5/transformer_encoder_15/Encoder-2nd-NormalizationLayer-1/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
Dmodel_5/transformer_encoder_15/Encoder-2nd-NormalizationLayer-1/ProdProdVmodel_5/transformer_encoder_15/Encoder-2nd-NormalizationLayer-1/strided_slice:output:0Nmodel_5/transformer_encoder_15/Encoder-2nd-NormalizationLayer-1/Const:output:0*
T0*
_output_shapes
: �
Umodel_5/transformer_encoder_15/Encoder-2nd-NormalizationLayer-1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:�
Wmodel_5/transformer_encoder_15/Encoder-2nd-NormalizationLayer-1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: �
Wmodel_5/transformer_encoder_15/Encoder-2nd-NormalizationLayer-1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
Omodel_5/transformer_encoder_15/Encoder-2nd-NormalizationLayer-1/strided_slice_1StridedSliceNmodel_5/transformer_encoder_15/Encoder-2nd-NormalizationLayer-1/Shape:output:0^model_5/transformer_encoder_15/Encoder-2nd-NormalizationLayer-1/strided_slice_1/stack:output:0`model_5/transformer_encoder_15/Encoder-2nd-NormalizationLayer-1/strided_slice_1/stack_1:output:0`model_5/transformer_encoder_15/Encoder-2nd-NormalizationLayer-1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask�
Gmodel_5/transformer_encoder_15/Encoder-2nd-NormalizationLayer-1/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
Fmodel_5/transformer_encoder_15/Encoder-2nd-NormalizationLayer-1/Prod_1ProdXmodel_5/transformer_encoder_15/Encoder-2nd-NormalizationLayer-1/strided_slice_1:output:0Pmodel_5/transformer_encoder_15/Encoder-2nd-NormalizationLayer-1/Const_1:output:0*
T0*
_output_shapes
: �
Omodel_5/transformer_encoder_15/Encoder-2nd-NormalizationLayer-1/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :�
Omodel_5/transformer_encoder_15/Encoder-2nd-NormalizationLayer-1/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :�
Mmodel_5/transformer_encoder_15/Encoder-2nd-NormalizationLayer-1/Reshape/shapePackXmodel_5/transformer_encoder_15/Encoder-2nd-NormalizationLayer-1/Reshape/shape/0:output:0Mmodel_5/transformer_encoder_15/Encoder-2nd-NormalizationLayer-1/Prod:output:0Omodel_5/transformer_encoder_15/Encoder-2nd-NormalizationLayer-1/Prod_1:output:0Xmodel_5/transformer_encoder_15/Encoder-2nd-NormalizationLayer-1/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:�
Gmodel_5/transformer_encoder_15/Encoder-2nd-NormalizationLayer-1/ReshapeReshapeBmodel_5/transformer_encoder_15/Encoder-1st-AdditionLayer-1/add:z:0Vmodel_5/transformer_encoder_15/Encoder-2nd-NormalizationLayer-1/Reshape/shape:output:0*
T0*8
_output_shapes&
$:"�������������������
Kmodel_5/transformer_encoder_15/Encoder-2nd-NormalizationLayer-1/ones/packedPackMmodel_5/transformer_encoder_15/Encoder-2nd-NormalizationLayer-1/Prod:output:0*
N*
T0*
_output_shapes
:�
Jmodel_5/transformer_encoder_15/Encoder-2nd-NormalizationLayer-1/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
Dmodel_5/transformer_encoder_15/Encoder-2nd-NormalizationLayer-1/onesFillTmodel_5/transformer_encoder_15/Encoder-2nd-NormalizationLayer-1/ones/packed:output:0Smodel_5/transformer_encoder_15/Encoder-2nd-NormalizationLayer-1/ones/Const:output:0*
T0*#
_output_shapes
:����������
Lmodel_5/transformer_encoder_15/Encoder-2nd-NormalizationLayer-1/zeros/packedPackMmodel_5/transformer_encoder_15/Encoder-2nd-NormalizationLayer-1/Prod:output:0*
N*
T0*
_output_shapes
:�
Kmodel_5/transformer_encoder_15/Encoder-2nd-NormalizationLayer-1/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
Emodel_5/transformer_encoder_15/Encoder-2nd-NormalizationLayer-1/zerosFillUmodel_5/transformer_encoder_15/Encoder-2nd-NormalizationLayer-1/zeros/packed:output:0Tmodel_5/transformer_encoder_15/Encoder-2nd-NormalizationLayer-1/zeros/Const:output:0*
T0*#
_output_shapes
:����������
Gmodel_5/transformer_encoder_15/Encoder-2nd-NormalizationLayer-1/Const_2Const*
_output_shapes
: *
dtype0*
valueB �
Gmodel_5/transformer_encoder_15/Encoder-2nd-NormalizationLayer-1/Const_3Const*
_output_shapes
: *
dtype0*
valueB �
Pmodel_5/transformer_encoder_15/Encoder-2nd-NormalizationLayer-1/FusedBatchNormV3FusedBatchNormV3Pmodel_5/transformer_encoder_15/Encoder-2nd-NormalizationLayer-1/Reshape:output:0Mmodel_5/transformer_encoder_15/Encoder-2nd-NormalizationLayer-1/ones:output:0Nmodel_5/transformer_encoder_15/Encoder-2nd-NormalizationLayer-1/zeros:output:0Pmodel_5/transformer_encoder_15/Encoder-2nd-NormalizationLayer-1/Const_2:output:0Pmodel_5/transformer_encoder_15/Encoder-2nd-NormalizationLayer-1/Const_3:output:0*
T0*
U0*x
_output_shapesf
d:"������������������:���������:���������:���������:���������:*
data_formatNCHW*
epsilon%o�:�
Imodel_5/transformer_encoder_15/Encoder-2nd-NormalizationLayer-1/Reshape_1ReshapeTmodel_5/transformer_encoder_15/Encoder-2nd-NormalizationLayer-1/FusedBatchNormV3:y:0Nmodel_5/transformer_encoder_15/Encoder-2nd-NormalizationLayer-1/Shape:output:0*
T0*+
_output_shapes
:���������P	�
Rmodel_5/transformer_encoder_15/Encoder-2nd-NormalizationLayer-1/mul/ReadVariableOpReadVariableOp[model_5_transformer_encoder_15_encoder_2nd_normalizationlayer_1_mul_readvariableop_resource*
_output_shapes
:	*
dtype0�
Cmodel_5/transformer_encoder_15/Encoder-2nd-NormalizationLayer-1/mulMulRmodel_5/transformer_encoder_15/Encoder-2nd-NormalizationLayer-1/Reshape_1:output:0Zmodel_5/transformer_encoder_15/Encoder-2nd-NormalizationLayer-1/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������P	�
Rmodel_5/transformer_encoder_15/Encoder-2nd-NormalizationLayer-1/add/ReadVariableOpReadVariableOp[model_5_transformer_encoder_15_encoder_2nd_normalizationlayer_1_add_readvariableop_resource*
_output_shapes
:	*
dtype0�
Cmodel_5/transformer_encoder_15/Encoder-2nd-NormalizationLayer-1/addAddV2Gmodel_5/transformer_encoder_15/Encoder-2nd-NormalizationLayer-1/mul:z:0Zmodel_5/transformer_encoder_15/Encoder-2nd-NormalizationLayer-1/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������P	�
Tmodel_5/transformer_encoder_15/Encoder-FeedForwardLayer_1_1/Tensordot/ReadVariableOpReadVariableOp]model_5_transformer_encoder_15_encoder_feedforwardlayer_1_1_tensordot_readvariableop_resource*
_output_shapes

:		*
dtype0�
Jmodel_5/transformer_encoder_15/Encoder-FeedForwardLayer_1_1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:�
Jmodel_5/transformer_encoder_15/Encoder-FeedForwardLayer_1_1/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       �
Kmodel_5/transformer_encoder_15/Encoder-FeedForwardLayer_1_1/Tensordot/ShapeShapeGmodel_5/transformer_encoder_15/Encoder-2nd-NormalizationLayer-1/add:z:0*
T0*
_output_shapes
::���
Smodel_5/transformer_encoder_15/Encoder-FeedForwardLayer_1_1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Nmodel_5/transformer_encoder_15/Encoder-FeedForwardLayer_1_1/Tensordot/GatherV2GatherV2Tmodel_5/transformer_encoder_15/Encoder-FeedForwardLayer_1_1/Tensordot/Shape:output:0Smodel_5/transformer_encoder_15/Encoder-FeedForwardLayer_1_1/Tensordot/free:output:0\model_5/transformer_encoder_15/Encoder-FeedForwardLayer_1_1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
Umodel_5/transformer_encoder_15/Encoder-FeedForwardLayer_1_1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Pmodel_5/transformer_encoder_15/Encoder-FeedForwardLayer_1_1/Tensordot/GatherV2_1GatherV2Tmodel_5/transformer_encoder_15/Encoder-FeedForwardLayer_1_1/Tensordot/Shape:output:0Smodel_5/transformer_encoder_15/Encoder-FeedForwardLayer_1_1/Tensordot/axes:output:0^model_5/transformer_encoder_15/Encoder-FeedForwardLayer_1_1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
Kmodel_5/transformer_encoder_15/Encoder-FeedForwardLayer_1_1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
Jmodel_5/transformer_encoder_15/Encoder-FeedForwardLayer_1_1/Tensordot/ProdProdWmodel_5/transformer_encoder_15/Encoder-FeedForwardLayer_1_1/Tensordot/GatherV2:output:0Tmodel_5/transformer_encoder_15/Encoder-FeedForwardLayer_1_1/Tensordot/Const:output:0*
T0*
_output_shapes
: �
Mmodel_5/transformer_encoder_15/Encoder-FeedForwardLayer_1_1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
Lmodel_5/transformer_encoder_15/Encoder-FeedForwardLayer_1_1/Tensordot/Prod_1ProdYmodel_5/transformer_encoder_15/Encoder-FeedForwardLayer_1_1/Tensordot/GatherV2_1:output:0Vmodel_5/transformer_encoder_15/Encoder-FeedForwardLayer_1_1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: �
Qmodel_5/transformer_encoder_15/Encoder-FeedForwardLayer_1_1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Lmodel_5/transformer_encoder_15/Encoder-FeedForwardLayer_1_1/Tensordot/concatConcatV2Smodel_5/transformer_encoder_15/Encoder-FeedForwardLayer_1_1/Tensordot/free:output:0Smodel_5/transformer_encoder_15/Encoder-FeedForwardLayer_1_1/Tensordot/axes:output:0Zmodel_5/transformer_encoder_15/Encoder-FeedForwardLayer_1_1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
Kmodel_5/transformer_encoder_15/Encoder-FeedForwardLayer_1_1/Tensordot/stackPackSmodel_5/transformer_encoder_15/Encoder-FeedForwardLayer_1_1/Tensordot/Prod:output:0Umodel_5/transformer_encoder_15/Encoder-FeedForwardLayer_1_1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
Omodel_5/transformer_encoder_15/Encoder-FeedForwardLayer_1_1/Tensordot/transpose	TransposeGmodel_5/transformer_encoder_15/Encoder-2nd-NormalizationLayer-1/add:z:0Umodel_5/transformer_encoder_15/Encoder-FeedForwardLayer_1_1/Tensordot/concat:output:0*
T0*+
_output_shapes
:���������P	�
Mmodel_5/transformer_encoder_15/Encoder-FeedForwardLayer_1_1/Tensordot/ReshapeReshapeSmodel_5/transformer_encoder_15/Encoder-FeedForwardLayer_1_1/Tensordot/transpose:y:0Tmodel_5/transformer_encoder_15/Encoder-FeedForwardLayer_1_1/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
Lmodel_5/transformer_encoder_15/Encoder-FeedForwardLayer_1_1/Tensordot/MatMulMatMulVmodel_5/transformer_encoder_15/Encoder-FeedForwardLayer_1_1/Tensordot/Reshape:output:0\model_5/transformer_encoder_15/Encoder-FeedForwardLayer_1_1/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������	�
Mmodel_5/transformer_encoder_15/Encoder-FeedForwardLayer_1_1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:	�
Smodel_5/transformer_encoder_15/Encoder-FeedForwardLayer_1_1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Nmodel_5/transformer_encoder_15/Encoder-FeedForwardLayer_1_1/Tensordot/concat_1ConcatV2Wmodel_5/transformer_encoder_15/Encoder-FeedForwardLayer_1_1/Tensordot/GatherV2:output:0Vmodel_5/transformer_encoder_15/Encoder-FeedForwardLayer_1_1/Tensordot/Const_2:output:0\model_5/transformer_encoder_15/Encoder-FeedForwardLayer_1_1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
Emodel_5/transformer_encoder_15/Encoder-FeedForwardLayer_1_1/TensordotReshapeVmodel_5/transformer_encoder_15/Encoder-FeedForwardLayer_1_1/Tensordot/MatMul:product:0Wmodel_5/transformer_encoder_15/Encoder-FeedForwardLayer_1_1/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������P	�
Rmodel_5/transformer_encoder_15/Encoder-FeedForwardLayer_1_1/BiasAdd/ReadVariableOpReadVariableOp[model_5_transformer_encoder_15_encoder_feedforwardlayer_1_1_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype0�
Cmodel_5/transformer_encoder_15/Encoder-FeedForwardLayer_1_1/BiasAddBiasAddNmodel_5/transformer_encoder_15/Encoder-FeedForwardLayer_1_1/Tensordot:output:0Zmodel_5/transformer_encoder_15/Encoder-FeedForwardLayer_1_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������P	�
Fmodel_5/transformer_encoder_15/Encoder-FeedForwardLayer_1_1/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
Dmodel_5/transformer_encoder_15/Encoder-FeedForwardLayer_1_1/Gelu/mulMulOmodel_5/transformer_encoder_15/Encoder-FeedForwardLayer_1_1/Gelu/mul/x:output:0Lmodel_5/transformer_encoder_15/Encoder-FeedForwardLayer_1_1/BiasAdd:output:0*
T0*+
_output_shapes
:���������P	�
Gmodel_5/transformer_encoder_15/Encoder-FeedForwardLayer_1_1/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *��?�
Hmodel_5/transformer_encoder_15/Encoder-FeedForwardLayer_1_1/Gelu/truedivRealDivLmodel_5/transformer_encoder_15/Encoder-FeedForwardLayer_1_1/BiasAdd:output:0Pmodel_5/transformer_encoder_15/Encoder-FeedForwardLayer_1_1/Gelu/Cast/x:output:0*
T0*+
_output_shapes
:���������P	�
Dmodel_5/transformer_encoder_15/Encoder-FeedForwardLayer_1_1/Gelu/ErfErfLmodel_5/transformer_encoder_15/Encoder-FeedForwardLayer_1_1/Gelu/truediv:z:0*
T0*+
_output_shapes
:���������P	�
Fmodel_5/transformer_encoder_15/Encoder-FeedForwardLayer_1_1/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
Dmodel_5/transformer_encoder_15/Encoder-FeedForwardLayer_1_1/Gelu/addAddV2Omodel_5/transformer_encoder_15/Encoder-FeedForwardLayer_1_1/Gelu/add/x:output:0Hmodel_5/transformer_encoder_15/Encoder-FeedForwardLayer_1_1/Gelu/Erf:y:0*
T0*+
_output_shapes
:���������P	�
Fmodel_5/transformer_encoder_15/Encoder-FeedForwardLayer_1_1/Gelu/mul_1MulHmodel_5/transformer_encoder_15/Encoder-FeedForwardLayer_1_1/Gelu/mul:z:0Hmodel_5/transformer_encoder_15/Encoder-FeedForwardLayer_1_1/Gelu/add:z:0*
T0*+
_output_shapes
:���������P	�
Tmodel_5/transformer_encoder_15/Encoder-FeedForwardLayer_2_1/Tensordot/ReadVariableOpReadVariableOp]model_5_transformer_encoder_15_encoder_feedforwardlayer_2_1_tensordot_readvariableop_resource*
_output_shapes

:		*
dtype0�
Jmodel_5/transformer_encoder_15/Encoder-FeedForwardLayer_2_1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:�
Jmodel_5/transformer_encoder_15/Encoder-FeedForwardLayer_2_1/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       �
Kmodel_5/transformer_encoder_15/Encoder-FeedForwardLayer_2_1/Tensordot/ShapeShapeJmodel_5/transformer_encoder_15/Encoder-FeedForwardLayer_1_1/Gelu/mul_1:z:0*
T0*
_output_shapes
::���
Smodel_5/transformer_encoder_15/Encoder-FeedForwardLayer_2_1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Nmodel_5/transformer_encoder_15/Encoder-FeedForwardLayer_2_1/Tensordot/GatherV2GatherV2Tmodel_5/transformer_encoder_15/Encoder-FeedForwardLayer_2_1/Tensordot/Shape:output:0Smodel_5/transformer_encoder_15/Encoder-FeedForwardLayer_2_1/Tensordot/free:output:0\model_5/transformer_encoder_15/Encoder-FeedForwardLayer_2_1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
Umodel_5/transformer_encoder_15/Encoder-FeedForwardLayer_2_1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Pmodel_5/transformer_encoder_15/Encoder-FeedForwardLayer_2_1/Tensordot/GatherV2_1GatherV2Tmodel_5/transformer_encoder_15/Encoder-FeedForwardLayer_2_1/Tensordot/Shape:output:0Smodel_5/transformer_encoder_15/Encoder-FeedForwardLayer_2_1/Tensordot/axes:output:0^model_5/transformer_encoder_15/Encoder-FeedForwardLayer_2_1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
Kmodel_5/transformer_encoder_15/Encoder-FeedForwardLayer_2_1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
Jmodel_5/transformer_encoder_15/Encoder-FeedForwardLayer_2_1/Tensordot/ProdProdWmodel_5/transformer_encoder_15/Encoder-FeedForwardLayer_2_1/Tensordot/GatherV2:output:0Tmodel_5/transformer_encoder_15/Encoder-FeedForwardLayer_2_1/Tensordot/Const:output:0*
T0*
_output_shapes
: �
Mmodel_5/transformer_encoder_15/Encoder-FeedForwardLayer_2_1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
Lmodel_5/transformer_encoder_15/Encoder-FeedForwardLayer_2_1/Tensordot/Prod_1ProdYmodel_5/transformer_encoder_15/Encoder-FeedForwardLayer_2_1/Tensordot/GatherV2_1:output:0Vmodel_5/transformer_encoder_15/Encoder-FeedForwardLayer_2_1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: �
Qmodel_5/transformer_encoder_15/Encoder-FeedForwardLayer_2_1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Lmodel_5/transformer_encoder_15/Encoder-FeedForwardLayer_2_1/Tensordot/concatConcatV2Smodel_5/transformer_encoder_15/Encoder-FeedForwardLayer_2_1/Tensordot/free:output:0Smodel_5/transformer_encoder_15/Encoder-FeedForwardLayer_2_1/Tensordot/axes:output:0Zmodel_5/transformer_encoder_15/Encoder-FeedForwardLayer_2_1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
Kmodel_5/transformer_encoder_15/Encoder-FeedForwardLayer_2_1/Tensordot/stackPackSmodel_5/transformer_encoder_15/Encoder-FeedForwardLayer_2_1/Tensordot/Prod:output:0Umodel_5/transformer_encoder_15/Encoder-FeedForwardLayer_2_1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
Omodel_5/transformer_encoder_15/Encoder-FeedForwardLayer_2_1/Tensordot/transpose	TransposeJmodel_5/transformer_encoder_15/Encoder-FeedForwardLayer_1_1/Gelu/mul_1:z:0Umodel_5/transformer_encoder_15/Encoder-FeedForwardLayer_2_1/Tensordot/concat:output:0*
T0*+
_output_shapes
:���������P	�
Mmodel_5/transformer_encoder_15/Encoder-FeedForwardLayer_2_1/Tensordot/ReshapeReshapeSmodel_5/transformer_encoder_15/Encoder-FeedForwardLayer_2_1/Tensordot/transpose:y:0Tmodel_5/transformer_encoder_15/Encoder-FeedForwardLayer_2_1/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
Lmodel_5/transformer_encoder_15/Encoder-FeedForwardLayer_2_1/Tensordot/MatMulMatMulVmodel_5/transformer_encoder_15/Encoder-FeedForwardLayer_2_1/Tensordot/Reshape:output:0\model_5/transformer_encoder_15/Encoder-FeedForwardLayer_2_1/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������	�
Mmodel_5/transformer_encoder_15/Encoder-FeedForwardLayer_2_1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:	�
Smodel_5/transformer_encoder_15/Encoder-FeedForwardLayer_2_1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Nmodel_5/transformer_encoder_15/Encoder-FeedForwardLayer_2_1/Tensordot/concat_1ConcatV2Wmodel_5/transformer_encoder_15/Encoder-FeedForwardLayer_2_1/Tensordot/GatherV2:output:0Vmodel_5/transformer_encoder_15/Encoder-FeedForwardLayer_2_1/Tensordot/Const_2:output:0\model_5/transformer_encoder_15/Encoder-FeedForwardLayer_2_1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
Emodel_5/transformer_encoder_15/Encoder-FeedForwardLayer_2_1/TensordotReshapeVmodel_5/transformer_encoder_15/Encoder-FeedForwardLayer_2_1/Tensordot/MatMul:product:0Wmodel_5/transformer_encoder_15/Encoder-FeedForwardLayer_2_1/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������P	�
Rmodel_5/transformer_encoder_15/Encoder-FeedForwardLayer_2_1/BiasAdd/ReadVariableOpReadVariableOp[model_5_transformer_encoder_15_encoder_feedforwardlayer_2_1_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype0�
Cmodel_5/transformer_encoder_15/Encoder-FeedForwardLayer_2_1/BiasAddBiasAddNmodel_5/transformer_encoder_15/Encoder-FeedForwardLayer_2_1/Tensordot:output:0Zmodel_5/transformer_encoder_15/Encoder-FeedForwardLayer_2_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������P	�
2model_5/transformer_encoder_15/dropout_15/IdentityIdentityLmodel_5/transformer_encoder_15/Encoder-FeedForwardLayer_2_1/BiasAdd:output:0*
T0*+
_output_shapes
:���������P	�
>model_5/transformer_encoder_15/Encoder-2nd-AdditionLayer-1/addAddV2Bmodel_5/transformer_encoder_15/Encoder-1st-AdditionLayer-1/add:z:0;model_5/transformer_encoder_15/dropout_15/Identity:output:0*
T0*+
_output_shapes
:���������P	�
Emodel_5/transformer_encoder_16/Encoder-1st-NormalizationLayer-2/ShapeShapeBmodel_5/transformer_encoder_15/Encoder-2nd-AdditionLayer-1/add:z:0*
T0*
_output_shapes
::���
Smodel_5/transformer_encoder_16/Encoder-1st-NormalizationLayer-2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
Umodel_5/transformer_encoder_16/Encoder-1st-NormalizationLayer-2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
Umodel_5/transformer_encoder_16/Encoder-1st-NormalizationLayer-2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
Mmodel_5/transformer_encoder_16/Encoder-1st-NormalizationLayer-2/strided_sliceStridedSliceNmodel_5/transformer_encoder_16/Encoder-1st-NormalizationLayer-2/Shape:output:0\model_5/transformer_encoder_16/Encoder-1st-NormalizationLayer-2/strided_slice/stack:output:0^model_5/transformer_encoder_16/Encoder-1st-NormalizationLayer-2/strided_slice/stack_1:output:0^model_5/transformer_encoder_16/Encoder-1st-NormalizationLayer-2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask�
Emodel_5/transformer_encoder_16/Encoder-1st-NormalizationLayer-2/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
Dmodel_5/transformer_encoder_16/Encoder-1st-NormalizationLayer-2/ProdProdVmodel_5/transformer_encoder_16/Encoder-1st-NormalizationLayer-2/strided_slice:output:0Nmodel_5/transformer_encoder_16/Encoder-1st-NormalizationLayer-2/Const:output:0*
T0*
_output_shapes
: �
Umodel_5/transformer_encoder_16/Encoder-1st-NormalizationLayer-2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:�
Wmodel_5/transformer_encoder_16/Encoder-1st-NormalizationLayer-2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: �
Wmodel_5/transformer_encoder_16/Encoder-1st-NormalizationLayer-2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
Omodel_5/transformer_encoder_16/Encoder-1st-NormalizationLayer-2/strided_slice_1StridedSliceNmodel_5/transformer_encoder_16/Encoder-1st-NormalizationLayer-2/Shape:output:0^model_5/transformer_encoder_16/Encoder-1st-NormalizationLayer-2/strided_slice_1/stack:output:0`model_5/transformer_encoder_16/Encoder-1st-NormalizationLayer-2/strided_slice_1/stack_1:output:0`model_5/transformer_encoder_16/Encoder-1st-NormalizationLayer-2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask�
Gmodel_5/transformer_encoder_16/Encoder-1st-NormalizationLayer-2/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
Fmodel_5/transformer_encoder_16/Encoder-1st-NormalizationLayer-2/Prod_1ProdXmodel_5/transformer_encoder_16/Encoder-1st-NormalizationLayer-2/strided_slice_1:output:0Pmodel_5/transformer_encoder_16/Encoder-1st-NormalizationLayer-2/Const_1:output:0*
T0*
_output_shapes
: �
Omodel_5/transformer_encoder_16/Encoder-1st-NormalizationLayer-2/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :�
Omodel_5/transformer_encoder_16/Encoder-1st-NormalizationLayer-2/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :�
Mmodel_5/transformer_encoder_16/Encoder-1st-NormalizationLayer-2/Reshape/shapePackXmodel_5/transformer_encoder_16/Encoder-1st-NormalizationLayer-2/Reshape/shape/0:output:0Mmodel_5/transformer_encoder_16/Encoder-1st-NormalizationLayer-2/Prod:output:0Omodel_5/transformer_encoder_16/Encoder-1st-NormalizationLayer-2/Prod_1:output:0Xmodel_5/transformer_encoder_16/Encoder-1st-NormalizationLayer-2/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:�
Gmodel_5/transformer_encoder_16/Encoder-1st-NormalizationLayer-2/ReshapeReshapeBmodel_5/transformer_encoder_15/Encoder-2nd-AdditionLayer-1/add:z:0Vmodel_5/transformer_encoder_16/Encoder-1st-NormalizationLayer-2/Reshape/shape:output:0*
T0*8
_output_shapes&
$:"�������������������
Kmodel_5/transformer_encoder_16/Encoder-1st-NormalizationLayer-2/ones/packedPackMmodel_5/transformer_encoder_16/Encoder-1st-NormalizationLayer-2/Prod:output:0*
N*
T0*
_output_shapes
:�
Jmodel_5/transformer_encoder_16/Encoder-1st-NormalizationLayer-2/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
Dmodel_5/transformer_encoder_16/Encoder-1st-NormalizationLayer-2/onesFillTmodel_5/transformer_encoder_16/Encoder-1st-NormalizationLayer-2/ones/packed:output:0Smodel_5/transformer_encoder_16/Encoder-1st-NormalizationLayer-2/ones/Const:output:0*
T0*#
_output_shapes
:����������
Lmodel_5/transformer_encoder_16/Encoder-1st-NormalizationLayer-2/zeros/packedPackMmodel_5/transformer_encoder_16/Encoder-1st-NormalizationLayer-2/Prod:output:0*
N*
T0*
_output_shapes
:�
Kmodel_5/transformer_encoder_16/Encoder-1st-NormalizationLayer-2/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
Emodel_5/transformer_encoder_16/Encoder-1st-NormalizationLayer-2/zerosFillUmodel_5/transformer_encoder_16/Encoder-1st-NormalizationLayer-2/zeros/packed:output:0Tmodel_5/transformer_encoder_16/Encoder-1st-NormalizationLayer-2/zeros/Const:output:0*
T0*#
_output_shapes
:����������
Gmodel_5/transformer_encoder_16/Encoder-1st-NormalizationLayer-2/Const_2Const*
_output_shapes
: *
dtype0*
valueB �
Gmodel_5/transformer_encoder_16/Encoder-1st-NormalizationLayer-2/Const_3Const*
_output_shapes
: *
dtype0*
valueB �
Pmodel_5/transformer_encoder_16/Encoder-1st-NormalizationLayer-2/FusedBatchNormV3FusedBatchNormV3Pmodel_5/transformer_encoder_16/Encoder-1st-NormalizationLayer-2/Reshape:output:0Mmodel_5/transformer_encoder_16/Encoder-1st-NormalizationLayer-2/ones:output:0Nmodel_5/transformer_encoder_16/Encoder-1st-NormalizationLayer-2/zeros:output:0Pmodel_5/transformer_encoder_16/Encoder-1st-NormalizationLayer-2/Const_2:output:0Pmodel_5/transformer_encoder_16/Encoder-1st-NormalizationLayer-2/Const_3:output:0*
T0*
U0*x
_output_shapesf
d:"������������������:���������:���������:���������:���������:*
data_formatNCHW*
epsilon%o�:�
Imodel_5/transformer_encoder_16/Encoder-1st-NormalizationLayer-2/Reshape_1ReshapeTmodel_5/transformer_encoder_16/Encoder-1st-NormalizationLayer-2/FusedBatchNormV3:y:0Nmodel_5/transformer_encoder_16/Encoder-1st-NormalizationLayer-2/Shape:output:0*
T0*+
_output_shapes
:���������P	�
Rmodel_5/transformer_encoder_16/Encoder-1st-NormalizationLayer-2/mul/ReadVariableOpReadVariableOp[model_5_transformer_encoder_16_encoder_1st_normalizationlayer_2_mul_readvariableop_resource*
_output_shapes
:	*
dtype0�
Cmodel_5/transformer_encoder_16/Encoder-1st-NormalizationLayer-2/mulMulRmodel_5/transformer_encoder_16/Encoder-1st-NormalizationLayer-2/Reshape_1:output:0Zmodel_5/transformer_encoder_16/Encoder-1st-NormalizationLayer-2/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������P	�
Rmodel_5/transformer_encoder_16/Encoder-1st-NormalizationLayer-2/add/ReadVariableOpReadVariableOp[model_5_transformer_encoder_16_encoder_1st_normalizationlayer_2_add_readvariableop_resource*
_output_shapes
:	*
dtype0�
Cmodel_5/transformer_encoder_16/Encoder-1st-NormalizationLayer-2/addAddV2Gmodel_5/transformer_encoder_16/Encoder-1st-NormalizationLayer-2/mul:z:0Zmodel_5/transformer_encoder_16/Encoder-1st-NormalizationLayer-2/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������P	�
^model_5/transformer_encoder_16/Encoder-SelfAttentionLayer-2/query/einsum/Einsum/ReadVariableOpReadVariableOpgmodel_5_transformer_encoder_16_encoder_selfattentionlayer_2_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:	*
dtype0�
Omodel_5/transformer_encoder_16/Encoder-SelfAttentionLayer-2/query/einsum/EinsumEinsumGmodel_5/transformer_encoder_16/Encoder-1st-NormalizationLayer-2/add:z:0fmodel_5/transformer_encoder_16/Encoder-SelfAttentionLayer-2/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������P*
equationabc,cde->abde�
Tmodel_5/transformer_encoder_16/Encoder-SelfAttentionLayer-2/query/add/ReadVariableOpReadVariableOp]model_5_transformer_encoder_16_encoder_selfattentionlayer_2_query_add_readvariableop_resource*
_output_shapes

:*
dtype0�
Emodel_5/transformer_encoder_16/Encoder-SelfAttentionLayer-2/query/addAddV2Xmodel_5/transformer_encoder_16/Encoder-SelfAttentionLayer-2/query/einsum/Einsum:output:0\model_5/transformer_encoder_16/Encoder-SelfAttentionLayer-2/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������P�
\model_5/transformer_encoder_16/Encoder-SelfAttentionLayer-2/key/einsum/Einsum/ReadVariableOpReadVariableOpemodel_5_transformer_encoder_16_encoder_selfattentionlayer_2_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:	*
dtype0�
Mmodel_5/transformer_encoder_16/Encoder-SelfAttentionLayer-2/key/einsum/EinsumEinsumGmodel_5/transformer_encoder_16/Encoder-1st-NormalizationLayer-2/add:z:0dmodel_5/transformer_encoder_16/Encoder-SelfAttentionLayer-2/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������P*
equationabc,cde->abde�
Rmodel_5/transformer_encoder_16/Encoder-SelfAttentionLayer-2/key/add/ReadVariableOpReadVariableOp[model_5_transformer_encoder_16_encoder_selfattentionlayer_2_key_add_readvariableop_resource*
_output_shapes

:*
dtype0�
Cmodel_5/transformer_encoder_16/Encoder-SelfAttentionLayer-2/key/addAddV2Vmodel_5/transformer_encoder_16/Encoder-SelfAttentionLayer-2/key/einsum/Einsum:output:0Zmodel_5/transformer_encoder_16/Encoder-SelfAttentionLayer-2/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������P�
^model_5/transformer_encoder_16/Encoder-SelfAttentionLayer-2/value/einsum/Einsum/ReadVariableOpReadVariableOpgmodel_5_transformer_encoder_16_encoder_selfattentionlayer_2_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:	*
dtype0�
Omodel_5/transformer_encoder_16/Encoder-SelfAttentionLayer-2/value/einsum/EinsumEinsumGmodel_5/transformer_encoder_16/Encoder-1st-NormalizationLayer-2/add:z:0fmodel_5/transformer_encoder_16/Encoder-SelfAttentionLayer-2/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������P*
equationabc,cde->abde�
Tmodel_5/transformer_encoder_16/Encoder-SelfAttentionLayer-2/value/add/ReadVariableOpReadVariableOp]model_5_transformer_encoder_16_encoder_selfattentionlayer_2_value_add_readvariableop_resource*
_output_shapes

:*
dtype0�
Emodel_5/transformer_encoder_16/Encoder-SelfAttentionLayer-2/value/addAddV2Xmodel_5/transformer_encoder_16/Encoder-SelfAttentionLayer-2/value/einsum/Einsum:output:0\model_5/transformer_encoder_16/Encoder-SelfAttentionLayer-2/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������P�
Amodel_5/transformer_encoder_16/Encoder-SelfAttentionLayer-2/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *:�?�
?model_5/transformer_encoder_16/Encoder-SelfAttentionLayer-2/MulMulImodel_5/transformer_encoder_16/Encoder-SelfAttentionLayer-2/query/add:z:0Jmodel_5/transformer_encoder_16/Encoder-SelfAttentionLayer-2/Mul/y:output:0*
T0*/
_output_shapes
:���������P�
Imodel_5/transformer_encoder_16/Encoder-SelfAttentionLayer-2/einsum/EinsumEinsumGmodel_5/transformer_encoder_16/Encoder-SelfAttentionLayer-2/key/add:z:0Cmodel_5/transformer_encoder_16/Encoder-SelfAttentionLayer-2/Mul:z:0*
N*
T0*/
_output_shapes
:���������PP*
equationaecd,abcd->acbe�
Kmodel_5/transformer_encoder_16/Encoder-SelfAttentionLayer-2/softmax/SoftmaxSoftmaxRmodel_5/transformer_encoder_16/Encoder-SelfAttentionLayer-2/einsum/Einsum:output:0*
T0*/
_output_shapes
:���������PP�
Lmodel_5/transformer_encoder_16/Encoder-SelfAttentionLayer-2/dropout/IdentityIdentityUmodel_5/transformer_encoder_16/Encoder-SelfAttentionLayer-2/softmax/Softmax:softmax:0*
T0*/
_output_shapes
:���������PP�
Kmodel_5/transformer_encoder_16/Encoder-SelfAttentionLayer-2/einsum_1/EinsumEinsumUmodel_5/transformer_encoder_16/Encoder-SelfAttentionLayer-2/dropout/Identity:output:0Imodel_5/transformer_encoder_16/Encoder-SelfAttentionLayer-2/value/add:z:0*
N*
T0*/
_output_shapes
:���������P*
equationacbe,aecd->abcd�
imodel_5/transformer_encoder_16/Encoder-SelfAttentionLayer-2/attention_output/einsum/Einsum/ReadVariableOpReadVariableOprmodel_5_transformer_encoder_16_encoder_selfattentionlayer_2_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:	*
dtype0�
Zmodel_5/transformer_encoder_16/Encoder-SelfAttentionLayer-2/attention_output/einsum/EinsumEinsumTmodel_5/transformer_encoder_16/Encoder-SelfAttentionLayer-2/einsum_1/Einsum:output:0qmodel_5/transformer_encoder_16/Encoder-SelfAttentionLayer-2/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:���������P	*
equationabcd,cde->abe�
_model_5/transformer_encoder_16/Encoder-SelfAttentionLayer-2/attention_output/add/ReadVariableOpReadVariableOphmodel_5_transformer_encoder_16_encoder_selfattentionlayer_2_attention_output_add_readvariableop_resource*
_output_shapes
:	*
dtype0�
Pmodel_5/transformer_encoder_16/Encoder-SelfAttentionLayer-2/attention_output/addAddV2cmodel_5/transformer_encoder_16/Encoder-SelfAttentionLayer-2/attention_output/einsum/Einsum:output:0gmodel_5/transformer_encoder_16/Encoder-SelfAttentionLayer-2/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������P	�
>model_5/transformer_encoder_16/Encoder-1st-AdditionLayer-2/addAddV2Bmodel_5/transformer_encoder_15/Encoder-2nd-AdditionLayer-1/add:z:0Tmodel_5/transformer_encoder_16/Encoder-SelfAttentionLayer-2/attention_output/add:z:0*
T0*+
_output_shapes
:���������P	�
Emodel_5/transformer_encoder_16/Encoder-2nd-NormalizationLayer-2/ShapeShapeBmodel_5/transformer_encoder_16/Encoder-1st-AdditionLayer-2/add:z:0*
T0*
_output_shapes
::���
Smodel_5/transformer_encoder_16/Encoder-2nd-NormalizationLayer-2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
Umodel_5/transformer_encoder_16/Encoder-2nd-NormalizationLayer-2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
Umodel_5/transformer_encoder_16/Encoder-2nd-NormalizationLayer-2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
Mmodel_5/transformer_encoder_16/Encoder-2nd-NormalizationLayer-2/strided_sliceStridedSliceNmodel_5/transformer_encoder_16/Encoder-2nd-NormalizationLayer-2/Shape:output:0\model_5/transformer_encoder_16/Encoder-2nd-NormalizationLayer-2/strided_slice/stack:output:0^model_5/transformer_encoder_16/Encoder-2nd-NormalizationLayer-2/strided_slice/stack_1:output:0^model_5/transformer_encoder_16/Encoder-2nd-NormalizationLayer-2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask�
Emodel_5/transformer_encoder_16/Encoder-2nd-NormalizationLayer-2/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
Dmodel_5/transformer_encoder_16/Encoder-2nd-NormalizationLayer-2/ProdProdVmodel_5/transformer_encoder_16/Encoder-2nd-NormalizationLayer-2/strided_slice:output:0Nmodel_5/transformer_encoder_16/Encoder-2nd-NormalizationLayer-2/Const:output:0*
T0*
_output_shapes
: �
Umodel_5/transformer_encoder_16/Encoder-2nd-NormalizationLayer-2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:�
Wmodel_5/transformer_encoder_16/Encoder-2nd-NormalizationLayer-2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: �
Wmodel_5/transformer_encoder_16/Encoder-2nd-NormalizationLayer-2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
Omodel_5/transformer_encoder_16/Encoder-2nd-NormalizationLayer-2/strided_slice_1StridedSliceNmodel_5/transformer_encoder_16/Encoder-2nd-NormalizationLayer-2/Shape:output:0^model_5/transformer_encoder_16/Encoder-2nd-NormalizationLayer-2/strided_slice_1/stack:output:0`model_5/transformer_encoder_16/Encoder-2nd-NormalizationLayer-2/strided_slice_1/stack_1:output:0`model_5/transformer_encoder_16/Encoder-2nd-NormalizationLayer-2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask�
Gmodel_5/transformer_encoder_16/Encoder-2nd-NormalizationLayer-2/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
Fmodel_5/transformer_encoder_16/Encoder-2nd-NormalizationLayer-2/Prod_1ProdXmodel_5/transformer_encoder_16/Encoder-2nd-NormalizationLayer-2/strided_slice_1:output:0Pmodel_5/transformer_encoder_16/Encoder-2nd-NormalizationLayer-2/Const_1:output:0*
T0*
_output_shapes
: �
Omodel_5/transformer_encoder_16/Encoder-2nd-NormalizationLayer-2/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :�
Omodel_5/transformer_encoder_16/Encoder-2nd-NormalizationLayer-2/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :�
Mmodel_5/transformer_encoder_16/Encoder-2nd-NormalizationLayer-2/Reshape/shapePackXmodel_5/transformer_encoder_16/Encoder-2nd-NormalizationLayer-2/Reshape/shape/0:output:0Mmodel_5/transformer_encoder_16/Encoder-2nd-NormalizationLayer-2/Prod:output:0Omodel_5/transformer_encoder_16/Encoder-2nd-NormalizationLayer-2/Prod_1:output:0Xmodel_5/transformer_encoder_16/Encoder-2nd-NormalizationLayer-2/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:�
Gmodel_5/transformer_encoder_16/Encoder-2nd-NormalizationLayer-2/ReshapeReshapeBmodel_5/transformer_encoder_16/Encoder-1st-AdditionLayer-2/add:z:0Vmodel_5/transformer_encoder_16/Encoder-2nd-NormalizationLayer-2/Reshape/shape:output:0*
T0*8
_output_shapes&
$:"�������������������
Kmodel_5/transformer_encoder_16/Encoder-2nd-NormalizationLayer-2/ones/packedPackMmodel_5/transformer_encoder_16/Encoder-2nd-NormalizationLayer-2/Prod:output:0*
N*
T0*
_output_shapes
:�
Jmodel_5/transformer_encoder_16/Encoder-2nd-NormalizationLayer-2/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
Dmodel_5/transformer_encoder_16/Encoder-2nd-NormalizationLayer-2/onesFillTmodel_5/transformer_encoder_16/Encoder-2nd-NormalizationLayer-2/ones/packed:output:0Smodel_5/transformer_encoder_16/Encoder-2nd-NormalizationLayer-2/ones/Const:output:0*
T0*#
_output_shapes
:����������
Lmodel_5/transformer_encoder_16/Encoder-2nd-NormalizationLayer-2/zeros/packedPackMmodel_5/transformer_encoder_16/Encoder-2nd-NormalizationLayer-2/Prod:output:0*
N*
T0*
_output_shapes
:�
Kmodel_5/transformer_encoder_16/Encoder-2nd-NormalizationLayer-2/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
Emodel_5/transformer_encoder_16/Encoder-2nd-NormalizationLayer-2/zerosFillUmodel_5/transformer_encoder_16/Encoder-2nd-NormalizationLayer-2/zeros/packed:output:0Tmodel_5/transformer_encoder_16/Encoder-2nd-NormalizationLayer-2/zeros/Const:output:0*
T0*#
_output_shapes
:����������
Gmodel_5/transformer_encoder_16/Encoder-2nd-NormalizationLayer-2/Const_2Const*
_output_shapes
: *
dtype0*
valueB �
Gmodel_5/transformer_encoder_16/Encoder-2nd-NormalizationLayer-2/Const_3Const*
_output_shapes
: *
dtype0*
valueB �
Pmodel_5/transformer_encoder_16/Encoder-2nd-NormalizationLayer-2/FusedBatchNormV3FusedBatchNormV3Pmodel_5/transformer_encoder_16/Encoder-2nd-NormalizationLayer-2/Reshape:output:0Mmodel_5/transformer_encoder_16/Encoder-2nd-NormalizationLayer-2/ones:output:0Nmodel_5/transformer_encoder_16/Encoder-2nd-NormalizationLayer-2/zeros:output:0Pmodel_5/transformer_encoder_16/Encoder-2nd-NormalizationLayer-2/Const_2:output:0Pmodel_5/transformer_encoder_16/Encoder-2nd-NormalizationLayer-2/Const_3:output:0*
T0*
U0*x
_output_shapesf
d:"������������������:���������:���������:���������:���������:*
data_formatNCHW*
epsilon%o�:�
Imodel_5/transformer_encoder_16/Encoder-2nd-NormalizationLayer-2/Reshape_1ReshapeTmodel_5/transformer_encoder_16/Encoder-2nd-NormalizationLayer-2/FusedBatchNormV3:y:0Nmodel_5/transformer_encoder_16/Encoder-2nd-NormalizationLayer-2/Shape:output:0*
T0*+
_output_shapes
:���������P	�
Rmodel_5/transformer_encoder_16/Encoder-2nd-NormalizationLayer-2/mul/ReadVariableOpReadVariableOp[model_5_transformer_encoder_16_encoder_2nd_normalizationlayer_2_mul_readvariableop_resource*
_output_shapes
:	*
dtype0�
Cmodel_5/transformer_encoder_16/Encoder-2nd-NormalizationLayer-2/mulMulRmodel_5/transformer_encoder_16/Encoder-2nd-NormalizationLayer-2/Reshape_1:output:0Zmodel_5/transformer_encoder_16/Encoder-2nd-NormalizationLayer-2/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������P	�
Rmodel_5/transformer_encoder_16/Encoder-2nd-NormalizationLayer-2/add/ReadVariableOpReadVariableOp[model_5_transformer_encoder_16_encoder_2nd_normalizationlayer_2_add_readvariableop_resource*
_output_shapes
:	*
dtype0�
Cmodel_5/transformer_encoder_16/Encoder-2nd-NormalizationLayer-2/addAddV2Gmodel_5/transformer_encoder_16/Encoder-2nd-NormalizationLayer-2/mul:z:0Zmodel_5/transformer_encoder_16/Encoder-2nd-NormalizationLayer-2/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������P	�
Tmodel_5/transformer_encoder_16/Encoder-FeedForwardLayer_1_2/Tensordot/ReadVariableOpReadVariableOp]model_5_transformer_encoder_16_encoder_feedforwardlayer_1_2_tensordot_readvariableop_resource*
_output_shapes

:		*
dtype0�
Jmodel_5/transformer_encoder_16/Encoder-FeedForwardLayer_1_2/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:�
Jmodel_5/transformer_encoder_16/Encoder-FeedForwardLayer_1_2/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       �
Kmodel_5/transformer_encoder_16/Encoder-FeedForwardLayer_1_2/Tensordot/ShapeShapeGmodel_5/transformer_encoder_16/Encoder-2nd-NormalizationLayer-2/add:z:0*
T0*
_output_shapes
::���
Smodel_5/transformer_encoder_16/Encoder-FeedForwardLayer_1_2/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Nmodel_5/transformer_encoder_16/Encoder-FeedForwardLayer_1_2/Tensordot/GatherV2GatherV2Tmodel_5/transformer_encoder_16/Encoder-FeedForwardLayer_1_2/Tensordot/Shape:output:0Smodel_5/transformer_encoder_16/Encoder-FeedForwardLayer_1_2/Tensordot/free:output:0\model_5/transformer_encoder_16/Encoder-FeedForwardLayer_1_2/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
Umodel_5/transformer_encoder_16/Encoder-FeedForwardLayer_1_2/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Pmodel_5/transformer_encoder_16/Encoder-FeedForwardLayer_1_2/Tensordot/GatherV2_1GatherV2Tmodel_5/transformer_encoder_16/Encoder-FeedForwardLayer_1_2/Tensordot/Shape:output:0Smodel_5/transformer_encoder_16/Encoder-FeedForwardLayer_1_2/Tensordot/axes:output:0^model_5/transformer_encoder_16/Encoder-FeedForwardLayer_1_2/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
Kmodel_5/transformer_encoder_16/Encoder-FeedForwardLayer_1_2/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
Jmodel_5/transformer_encoder_16/Encoder-FeedForwardLayer_1_2/Tensordot/ProdProdWmodel_5/transformer_encoder_16/Encoder-FeedForwardLayer_1_2/Tensordot/GatherV2:output:0Tmodel_5/transformer_encoder_16/Encoder-FeedForwardLayer_1_2/Tensordot/Const:output:0*
T0*
_output_shapes
: �
Mmodel_5/transformer_encoder_16/Encoder-FeedForwardLayer_1_2/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
Lmodel_5/transformer_encoder_16/Encoder-FeedForwardLayer_1_2/Tensordot/Prod_1ProdYmodel_5/transformer_encoder_16/Encoder-FeedForwardLayer_1_2/Tensordot/GatherV2_1:output:0Vmodel_5/transformer_encoder_16/Encoder-FeedForwardLayer_1_2/Tensordot/Const_1:output:0*
T0*
_output_shapes
: �
Qmodel_5/transformer_encoder_16/Encoder-FeedForwardLayer_1_2/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Lmodel_5/transformer_encoder_16/Encoder-FeedForwardLayer_1_2/Tensordot/concatConcatV2Smodel_5/transformer_encoder_16/Encoder-FeedForwardLayer_1_2/Tensordot/free:output:0Smodel_5/transformer_encoder_16/Encoder-FeedForwardLayer_1_2/Tensordot/axes:output:0Zmodel_5/transformer_encoder_16/Encoder-FeedForwardLayer_1_2/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
Kmodel_5/transformer_encoder_16/Encoder-FeedForwardLayer_1_2/Tensordot/stackPackSmodel_5/transformer_encoder_16/Encoder-FeedForwardLayer_1_2/Tensordot/Prod:output:0Umodel_5/transformer_encoder_16/Encoder-FeedForwardLayer_1_2/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
Omodel_5/transformer_encoder_16/Encoder-FeedForwardLayer_1_2/Tensordot/transpose	TransposeGmodel_5/transformer_encoder_16/Encoder-2nd-NormalizationLayer-2/add:z:0Umodel_5/transformer_encoder_16/Encoder-FeedForwardLayer_1_2/Tensordot/concat:output:0*
T0*+
_output_shapes
:���������P	�
Mmodel_5/transformer_encoder_16/Encoder-FeedForwardLayer_1_2/Tensordot/ReshapeReshapeSmodel_5/transformer_encoder_16/Encoder-FeedForwardLayer_1_2/Tensordot/transpose:y:0Tmodel_5/transformer_encoder_16/Encoder-FeedForwardLayer_1_2/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
Lmodel_5/transformer_encoder_16/Encoder-FeedForwardLayer_1_2/Tensordot/MatMulMatMulVmodel_5/transformer_encoder_16/Encoder-FeedForwardLayer_1_2/Tensordot/Reshape:output:0\model_5/transformer_encoder_16/Encoder-FeedForwardLayer_1_2/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������	�
Mmodel_5/transformer_encoder_16/Encoder-FeedForwardLayer_1_2/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:	�
Smodel_5/transformer_encoder_16/Encoder-FeedForwardLayer_1_2/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Nmodel_5/transformer_encoder_16/Encoder-FeedForwardLayer_1_2/Tensordot/concat_1ConcatV2Wmodel_5/transformer_encoder_16/Encoder-FeedForwardLayer_1_2/Tensordot/GatherV2:output:0Vmodel_5/transformer_encoder_16/Encoder-FeedForwardLayer_1_2/Tensordot/Const_2:output:0\model_5/transformer_encoder_16/Encoder-FeedForwardLayer_1_2/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
Emodel_5/transformer_encoder_16/Encoder-FeedForwardLayer_1_2/TensordotReshapeVmodel_5/transformer_encoder_16/Encoder-FeedForwardLayer_1_2/Tensordot/MatMul:product:0Wmodel_5/transformer_encoder_16/Encoder-FeedForwardLayer_1_2/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������P	�
Rmodel_5/transformer_encoder_16/Encoder-FeedForwardLayer_1_2/BiasAdd/ReadVariableOpReadVariableOp[model_5_transformer_encoder_16_encoder_feedforwardlayer_1_2_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype0�
Cmodel_5/transformer_encoder_16/Encoder-FeedForwardLayer_1_2/BiasAddBiasAddNmodel_5/transformer_encoder_16/Encoder-FeedForwardLayer_1_2/Tensordot:output:0Zmodel_5/transformer_encoder_16/Encoder-FeedForwardLayer_1_2/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������P	�
Fmodel_5/transformer_encoder_16/Encoder-FeedForwardLayer_1_2/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
Dmodel_5/transformer_encoder_16/Encoder-FeedForwardLayer_1_2/Gelu/mulMulOmodel_5/transformer_encoder_16/Encoder-FeedForwardLayer_1_2/Gelu/mul/x:output:0Lmodel_5/transformer_encoder_16/Encoder-FeedForwardLayer_1_2/BiasAdd:output:0*
T0*+
_output_shapes
:���������P	�
Gmodel_5/transformer_encoder_16/Encoder-FeedForwardLayer_1_2/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *��?�
Hmodel_5/transformer_encoder_16/Encoder-FeedForwardLayer_1_2/Gelu/truedivRealDivLmodel_5/transformer_encoder_16/Encoder-FeedForwardLayer_1_2/BiasAdd:output:0Pmodel_5/transformer_encoder_16/Encoder-FeedForwardLayer_1_2/Gelu/Cast/x:output:0*
T0*+
_output_shapes
:���������P	�
Dmodel_5/transformer_encoder_16/Encoder-FeedForwardLayer_1_2/Gelu/ErfErfLmodel_5/transformer_encoder_16/Encoder-FeedForwardLayer_1_2/Gelu/truediv:z:0*
T0*+
_output_shapes
:���������P	�
Fmodel_5/transformer_encoder_16/Encoder-FeedForwardLayer_1_2/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
Dmodel_5/transformer_encoder_16/Encoder-FeedForwardLayer_1_2/Gelu/addAddV2Omodel_5/transformer_encoder_16/Encoder-FeedForwardLayer_1_2/Gelu/add/x:output:0Hmodel_5/transformer_encoder_16/Encoder-FeedForwardLayer_1_2/Gelu/Erf:y:0*
T0*+
_output_shapes
:���������P	�
Fmodel_5/transformer_encoder_16/Encoder-FeedForwardLayer_1_2/Gelu/mul_1MulHmodel_5/transformer_encoder_16/Encoder-FeedForwardLayer_1_2/Gelu/mul:z:0Hmodel_5/transformer_encoder_16/Encoder-FeedForwardLayer_1_2/Gelu/add:z:0*
T0*+
_output_shapes
:���������P	�
Tmodel_5/transformer_encoder_16/Encoder-FeedForwardLayer_2_2/Tensordot/ReadVariableOpReadVariableOp]model_5_transformer_encoder_16_encoder_feedforwardlayer_2_2_tensordot_readvariableop_resource*
_output_shapes

:		*
dtype0�
Jmodel_5/transformer_encoder_16/Encoder-FeedForwardLayer_2_2/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:�
Jmodel_5/transformer_encoder_16/Encoder-FeedForwardLayer_2_2/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       �
Kmodel_5/transformer_encoder_16/Encoder-FeedForwardLayer_2_2/Tensordot/ShapeShapeJmodel_5/transformer_encoder_16/Encoder-FeedForwardLayer_1_2/Gelu/mul_1:z:0*
T0*
_output_shapes
::���
Smodel_5/transformer_encoder_16/Encoder-FeedForwardLayer_2_2/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Nmodel_5/transformer_encoder_16/Encoder-FeedForwardLayer_2_2/Tensordot/GatherV2GatherV2Tmodel_5/transformer_encoder_16/Encoder-FeedForwardLayer_2_2/Tensordot/Shape:output:0Smodel_5/transformer_encoder_16/Encoder-FeedForwardLayer_2_2/Tensordot/free:output:0\model_5/transformer_encoder_16/Encoder-FeedForwardLayer_2_2/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
Umodel_5/transformer_encoder_16/Encoder-FeedForwardLayer_2_2/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Pmodel_5/transformer_encoder_16/Encoder-FeedForwardLayer_2_2/Tensordot/GatherV2_1GatherV2Tmodel_5/transformer_encoder_16/Encoder-FeedForwardLayer_2_2/Tensordot/Shape:output:0Smodel_5/transformer_encoder_16/Encoder-FeedForwardLayer_2_2/Tensordot/axes:output:0^model_5/transformer_encoder_16/Encoder-FeedForwardLayer_2_2/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
Kmodel_5/transformer_encoder_16/Encoder-FeedForwardLayer_2_2/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
Jmodel_5/transformer_encoder_16/Encoder-FeedForwardLayer_2_2/Tensordot/ProdProdWmodel_5/transformer_encoder_16/Encoder-FeedForwardLayer_2_2/Tensordot/GatherV2:output:0Tmodel_5/transformer_encoder_16/Encoder-FeedForwardLayer_2_2/Tensordot/Const:output:0*
T0*
_output_shapes
: �
Mmodel_5/transformer_encoder_16/Encoder-FeedForwardLayer_2_2/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
Lmodel_5/transformer_encoder_16/Encoder-FeedForwardLayer_2_2/Tensordot/Prod_1ProdYmodel_5/transformer_encoder_16/Encoder-FeedForwardLayer_2_2/Tensordot/GatherV2_1:output:0Vmodel_5/transformer_encoder_16/Encoder-FeedForwardLayer_2_2/Tensordot/Const_1:output:0*
T0*
_output_shapes
: �
Qmodel_5/transformer_encoder_16/Encoder-FeedForwardLayer_2_2/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Lmodel_5/transformer_encoder_16/Encoder-FeedForwardLayer_2_2/Tensordot/concatConcatV2Smodel_5/transformer_encoder_16/Encoder-FeedForwardLayer_2_2/Tensordot/free:output:0Smodel_5/transformer_encoder_16/Encoder-FeedForwardLayer_2_2/Tensordot/axes:output:0Zmodel_5/transformer_encoder_16/Encoder-FeedForwardLayer_2_2/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
Kmodel_5/transformer_encoder_16/Encoder-FeedForwardLayer_2_2/Tensordot/stackPackSmodel_5/transformer_encoder_16/Encoder-FeedForwardLayer_2_2/Tensordot/Prod:output:0Umodel_5/transformer_encoder_16/Encoder-FeedForwardLayer_2_2/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
Omodel_5/transformer_encoder_16/Encoder-FeedForwardLayer_2_2/Tensordot/transpose	TransposeJmodel_5/transformer_encoder_16/Encoder-FeedForwardLayer_1_2/Gelu/mul_1:z:0Umodel_5/transformer_encoder_16/Encoder-FeedForwardLayer_2_2/Tensordot/concat:output:0*
T0*+
_output_shapes
:���������P	�
Mmodel_5/transformer_encoder_16/Encoder-FeedForwardLayer_2_2/Tensordot/ReshapeReshapeSmodel_5/transformer_encoder_16/Encoder-FeedForwardLayer_2_2/Tensordot/transpose:y:0Tmodel_5/transformer_encoder_16/Encoder-FeedForwardLayer_2_2/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
Lmodel_5/transformer_encoder_16/Encoder-FeedForwardLayer_2_2/Tensordot/MatMulMatMulVmodel_5/transformer_encoder_16/Encoder-FeedForwardLayer_2_2/Tensordot/Reshape:output:0\model_5/transformer_encoder_16/Encoder-FeedForwardLayer_2_2/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������	�
Mmodel_5/transformer_encoder_16/Encoder-FeedForwardLayer_2_2/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:	�
Smodel_5/transformer_encoder_16/Encoder-FeedForwardLayer_2_2/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Nmodel_5/transformer_encoder_16/Encoder-FeedForwardLayer_2_2/Tensordot/concat_1ConcatV2Wmodel_5/transformer_encoder_16/Encoder-FeedForwardLayer_2_2/Tensordot/GatherV2:output:0Vmodel_5/transformer_encoder_16/Encoder-FeedForwardLayer_2_2/Tensordot/Const_2:output:0\model_5/transformer_encoder_16/Encoder-FeedForwardLayer_2_2/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
Emodel_5/transformer_encoder_16/Encoder-FeedForwardLayer_2_2/TensordotReshapeVmodel_5/transformer_encoder_16/Encoder-FeedForwardLayer_2_2/Tensordot/MatMul:product:0Wmodel_5/transformer_encoder_16/Encoder-FeedForwardLayer_2_2/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������P	�
Rmodel_5/transformer_encoder_16/Encoder-FeedForwardLayer_2_2/BiasAdd/ReadVariableOpReadVariableOp[model_5_transformer_encoder_16_encoder_feedforwardlayer_2_2_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype0�
Cmodel_5/transformer_encoder_16/Encoder-FeedForwardLayer_2_2/BiasAddBiasAddNmodel_5/transformer_encoder_16/Encoder-FeedForwardLayer_2_2/Tensordot:output:0Zmodel_5/transformer_encoder_16/Encoder-FeedForwardLayer_2_2/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������P	�
2model_5/transformer_encoder_16/dropout_16/IdentityIdentityLmodel_5/transformer_encoder_16/Encoder-FeedForwardLayer_2_2/BiasAdd:output:0*
T0*+
_output_shapes
:���������P	�
>model_5/transformer_encoder_16/Encoder-2nd-AdditionLayer-2/addAddV2Bmodel_5/transformer_encoder_16/Encoder-1st-AdditionLayer-2/add:z:0;model_5/transformer_encoder_16/dropout_16/Identity:output:0*
T0*+
_output_shapes
:���������P	�
Emodel_5/transformer_encoder_17/Encoder-1st-NormalizationLayer-3/ShapeShapeBmodel_5/transformer_encoder_16/Encoder-2nd-AdditionLayer-2/add:z:0*
T0*
_output_shapes
::���
Smodel_5/transformer_encoder_17/Encoder-1st-NormalizationLayer-3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
Umodel_5/transformer_encoder_17/Encoder-1st-NormalizationLayer-3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
Umodel_5/transformer_encoder_17/Encoder-1st-NormalizationLayer-3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
Mmodel_5/transformer_encoder_17/Encoder-1st-NormalizationLayer-3/strided_sliceStridedSliceNmodel_5/transformer_encoder_17/Encoder-1st-NormalizationLayer-3/Shape:output:0\model_5/transformer_encoder_17/Encoder-1st-NormalizationLayer-3/strided_slice/stack:output:0^model_5/transformer_encoder_17/Encoder-1st-NormalizationLayer-3/strided_slice/stack_1:output:0^model_5/transformer_encoder_17/Encoder-1st-NormalizationLayer-3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask�
Emodel_5/transformer_encoder_17/Encoder-1st-NormalizationLayer-3/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
Dmodel_5/transformer_encoder_17/Encoder-1st-NormalizationLayer-3/ProdProdVmodel_5/transformer_encoder_17/Encoder-1st-NormalizationLayer-3/strided_slice:output:0Nmodel_5/transformer_encoder_17/Encoder-1st-NormalizationLayer-3/Const:output:0*
T0*
_output_shapes
: �
Umodel_5/transformer_encoder_17/Encoder-1st-NormalizationLayer-3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:�
Wmodel_5/transformer_encoder_17/Encoder-1st-NormalizationLayer-3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: �
Wmodel_5/transformer_encoder_17/Encoder-1st-NormalizationLayer-3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
Omodel_5/transformer_encoder_17/Encoder-1st-NormalizationLayer-3/strided_slice_1StridedSliceNmodel_5/transformer_encoder_17/Encoder-1st-NormalizationLayer-3/Shape:output:0^model_5/transformer_encoder_17/Encoder-1st-NormalizationLayer-3/strided_slice_1/stack:output:0`model_5/transformer_encoder_17/Encoder-1st-NormalizationLayer-3/strided_slice_1/stack_1:output:0`model_5/transformer_encoder_17/Encoder-1st-NormalizationLayer-3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask�
Gmodel_5/transformer_encoder_17/Encoder-1st-NormalizationLayer-3/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
Fmodel_5/transformer_encoder_17/Encoder-1st-NormalizationLayer-3/Prod_1ProdXmodel_5/transformer_encoder_17/Encoder-1st-NormalizationLayer-3/strided_slice_1:output:0Pmodel_5/transformer_encoder_17/Encoder-1st-NormalizationLayer-3/Const_1:output:0*
T0*
_output_shapes
: �
Omodel_5/transformer_encoder_17/Encoder-1st-NormalizationLayer-3/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :�
Omodel_5/transformer_encoder_17/Encoder-1st-NormalizationLayer-3/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :�
Mmodel_5/transformer_encoder_17/Encoder-1st-NormalizationLayer-3/Reshape/shapePackXmodel_5/transformer_encoder_17/Encoder-1st-NormalizationLayer-3/Reshape/shape/0:output:0Mmodel_5/transformer_encoder_17/Encoder-1st-NormalizationLayer-3/Prod:output:0Omodel_5/transformer_encoder_17/Encoder-1st-NormalizationLayer-3/Prod_1:output:0Xmodel_5/transformer_encoder_17/Encoder-1st-NormalizationLayer-3/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:�
Gmodel_5/transformer_encoder_17/Encoder-1st-NormalizationLayer-3/ReshapeReshapeBmodel_5/transformer_encoder_16/Encoder-2nd-AdditionLayer-2/add:z:0Vmodel_5/transformer_encoder_17/Encoder-1st-NormalizationLayer-3/Reshape/shape:output:0*
T0*8
_output_shapes&
$:"�������������������
Kmodel_5/transformer_encoder_17/Encoder-1st-NormalizationLayer-3/ones/packedPackMmodel_5/transformer_encoder_17/Encoder-1st-NormalizationLayer-3/Prod:output:0*
N*
T0*
_output_shapes
:�
Jmodel_5/transformer_encoder_17/Encoder-1st-NormalizationLayer-3/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
Dmodel_5/transformer_encoder_17/Encoder-1st-NormalizationLayer-3/onesFillTmodel_5/transformer_encoder_17/Encoder-1st-NormalizationLayer-3/ones/packed:output:0Smodel_5/transformer_encoder_17/Encoder-1st-NormalizationLayer-3/ones/Const:output:0*
T0*#
_output_shapes
:����������
Lmodel_5/transformer_encoder_17/Encoder-1st-NormalizationLayer-3/zeros/packedPackMmodel_5/transformer_encoder_17/Encoder-1st-NormalizationLayer-3/Prod:output:0*
N*
T0*
_output_shapes
:�
Kmodel_5/transformer_encoder_17/Encoder-1st-NormalizationLayer-3/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
Emodel_5/transformer_encoder_17/Encoder-1st-NormalizationLayer-3/zerosFillUmodel_5/transformer_encoder_17/Encoder-1st-NormalizationLayer-3/zeros/packed:output:0Tmodel_5/transformer_encoder_17/Encoder-1st-NormalizationLayer-3/zeros/Const:output:0*
T0*#
_output_shapes
:����������
Gmodel_5/transformer_encoder_17/Encoder-1st-NormalizationLayer-3/Const_2Const*
_output_shapes
: *
dtype0*
valueB �
Gmodel_5/transformer_encoder_17/Encoder-1st-NormalizationLayer-3/Const_3Const*
_output_shapes
: *
dtype0*
valueB �
Pmodel_5/transformer_encoder_17/Encoder-1st-NormalizationLayer-3/FusedBatchNormV3FusedBatchNormV3Pmodel_5/transformer_encoder_17/Encoder-1st-NormalizationLayer-3/Reshape:output:0Mmodel_5/transformer_encoder_17/Encoder-1st-NormalizationLayer-3/ones:output:0Nmodel_5/transformer_encoder_17/Encoder-1st-NormalizationLayer-3/zeros:output:0Pmodel_5/transformer_encoder_17/Encoder-1st-NormalizationLayer-3/Const_2:output:0Pmodel_5/transformer_encoder_17/Encoder-1st-NormalizationLayer-3/Const_3:output:0*
T0*
U0*x
_output_shapesf
d:"������������������:���������:���������:���������:���������:*
data_formatNCHW*
epsilon%o�:�
Imodel_5/transformer_encoder_17/Encoder-1st-NormalizationLayer-3/Reshape_1ReshapeTmodel_5/transformer_encoder_17/Encoder-1st-NormalizationLayer-3/FusedBatchNormV3:y:0Nmodel_5/transformer_encoder_17/Encoder-1st-NormalizationLayer-3/Shape:output:0*
T0*+
_output_shapes
:���������P	�
Rmodel_5/transformer_encoder_17/Encoder-1st-NormalizationLayer-3/mul/ReadVariableOpReadVariableOp[model_5_transformer_encoder_17_encoder_1st_normalizationlayer_3_mul_readvariableop_resource*
_output_shapes
:	*
dtype0�
Cmodel_5/transformer_encoder_17/Encoder-1st-NormalizationLayer-3/mulMulRmodel_5/transformer_encoder_17/Encoder-1st-NormalizationLayer-3/Reshape_1:output:0Zmodel_5/transformer_encoder_17/Encoder-1st-NormalizationLayer-3/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������P	�
Rmodel_5/transformer_encoder_17/Encoder-1st-NormalizationLayer-3/add/ReadVariableOpReadVariableOp[model_5_transformer_encoder_17_encoder_1st_normalizationlayer_3_add_readvariableop_resource*
_output_shapes
:	*
dtype0�
Cmodel_5/transformer_encoder_17/Encoder-1st-NormalizationLayer-3/addAddV2Gmodel_5/transformer_encoder_17/Encoder-1st-NormalizationLayer-3/mul:z:0Zmodel_5/transformer_encoder_17/Encoder-1st-NormalizationLayer-3/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������P	�
^model_5/transformer_encoder_17/Encoder-SelfAttentionLayer-3/query/einsum/Einsum/ReadVariableOpReadVariableOpgmodel_5_transformer_encoder_17_encoder_selfattentionlayer_3_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:	*
dtype0�
Omodel_5/transformer_encoder_17/Encoder-SelfAttentionLayer-3/query/einsum/EinsumEinsumGmodel_5/transformer_encoder_17/Encoder-1st-NormalizationLayer-3/add:z:0fmodel_5/transformer_encoder_17/Encoder-SelfAttentionLayer-3/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������P*
equationabc,cde->abde�
Tmodel_5/transformer_encoder_17/Encoder-SelfAttentionLayer-3/query/add/ReadVariableOpReadVariableOp]model_5_transformer_encoder_17_encoder_selfattentionlayer_3_query_add_readvariableop_resource*
_output_shapes

:*
dtype0�
Emodel_5/transformer_encoder_17/Encoder-SelfAttentionLayer-3/query/addAddV2Xmodel_5/transformer_encoder_17/Encoder-SelfAttentionLayer-3/query/einsum/Einsum:output:0\model_5/transformer_encoder_17/Encoder-SelfAttentionLayer-3/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������P�
\model_5/transformer_encoder_17/Encoder-SelfAttentionLayer-3/key/einsum/Einsum/ReadVariableOpReadVariableOpemodel_5_transformer_encoder_17_encoder_selfattentionlayer_3_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:	*
dtype0�
Mmodel_5/transformer_encoder_17/Encoder-SelfAttentionLayer-3/key/einsum/EinsumEinsumGmodel_5/transformer_encoder_17/Encoder-1st-NormalizationLayer-3/add:z:0dmodel_5/transformer_encoder_17/Encoder-SelfAttentionLayer-3/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������P*
equationabc,cde->abde�
Rmodel_5/transformer_encoder_17/Encoder-SelfAttentionLayer-3/key/add/ReadVariableOpReadVariableOp[model_5_transformer_encoder_17_encoder_selfattentionlayer_3_key_add_readvariableop_resource*
_output_shapes

:*
dtype0�
Cmodel_5/transformer_encoder_17/Encoder-SelfAttentionLayer-3/key/addAddV2Vmodel_5/transformer_encoder_17/Encoder-SelfAttentionLayer-3/key/einsum/Einsum:output:0Zmodel_5/transformer_encoder_17/Encoder-SelfAttentionLayer-3/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������P�
^model_5/transformer_encoder_17/Encoder-SelfAttentionLayer-3/value/einsum/Einsum/ReadVariableOpReadVariableOpgmodel_5_transformer_encoder_17_encoder_selfattentionlayer_3_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:	*
dtype0�
Omodel_5/transformer_encoder_17/Encoder-SelfAttentionLayer-3/value/einsum/EinsumEinsumGmodel_5/transformer_encoder_17/Encoder-1st-NormalizationLayer-3/add:z:0fmodel_5/transformer_encoder_17/Encoder-SelfAttentionLayer-3/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������P*
equationabc,cde->abde�
Tmodel_5/transformer_encoder_17/Encoder-SelfAttentionLayer-3/value/add/ReadVariableOpReadVariableOp]model_5_transformer_encoder_17_encoder_selfattentionlayer_3_value_add_readvariableop_resource*
_output_shapes

:*
dtype0�
Emodel_5/transformer_encoder_17/Encoder-SelfAttentionLayer-3/value/addAddV2Xmodel_5/transformer_encoder_17/Encoder-SelfAttentionLayer-3/value/einsum/Einsum:output:0\model_5/transformer_encoder_17/Encoder-SelfAttentionLayer-3/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������P�
Amodel_5/transformer_encoder_17/Encoder-SelfAttentionLayer-3/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *:�?�
?model_5/transformer_encoder_17/Encoder-SelfAttentionLayer-3/MulMulImodel_5/transformer_encoder_17/Encoder-SelfAttentionLayer-3/query/add:z:0Jmodel_5/transformer_encoder_17/Encoder-SelfAttentionLayer-3/Mul/y:output:0*
T0*/
_output_shapes
:���������P�
Imodel_5/transformer_encoder_17/Encoder-SelfAttentionLayer-3/einsum/EinsumEinsumGmodel_5/transformer_encoder_17/Encoder-SelfAttentionLayer-3/key/add:z:0Cmodel_5/transformer_encoder_17/Encoder-SelfAttentionLayer-3/Mul:z:0*
N*
T0*/
_output_shapes
:���������PP*
equationaecd,abcd->acbe�
Kmodel_5/transformer_encoder_17/Encoder-SelfAttentionLayer-3/softmax/SoftmaxSoftmaxRmodel_5/transformer_encoder_17/Encoder-SelfAttentionLayer-3/einsum/Einsum:output:0*
T0*/
_output_shapes
:���������PP�
Lmodel_5/transformer_encoder_17/Encoder-SelfAttentionLayer-3/dropout/IdentityIdentityUmodel_5/transformer_encoder_17/Encoder-SelfAttentionLayer-3/softmax/Softmax:softmax:0*
T0*/
_output_shapes
:���������PP�
Kmodel_5/transformer_encoder_17/Encoder-SelfAttentionLayer-3/einsum_1/EinsumEinsumUmodel_5/transformer_encoder_17/Encoder-SelfAttentionLayer-3/dropout/Identity:output:0Imodel_5/transformer_encoder_17/Encoder-SelfAttentionLayer-3/value/add:z:0*
N*
T0*/
_output_shapes
:���������P*
equationacbe,aecd->abcd�
imodel_5/transformer_encoder_17/Encoder-SelfAttentionLayer-3/attention_output/einsum/Einsum/ReadVariableOpReadVariableOprmodel_5_transformer_encoder_17_encoder_selfattentionlayer_3_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:	*
dtype0�
Zmodel_5/transformer_encoder_17/Encoder-SelfAttentionLayer-3/attention_output/einsum/EinsumEinsumTmodel_5/transformer_encoder_17/Encoder-SelfAttentionLayer-3/einsum_1/Einsum:output:0qmodel_5/transformer_encoder_17/Encoder-SelfAttentionLayer-3/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:���������P	*
equationabcd,cde->abe�
_model_5/transformer_encoder_17/Encoder-SelfAttentionLayer-3/attention_output/add/ReadVariableOpReadVariableOphmodel_5_transformer_encoder_17_encoder_selfattentionlayer_3_attention_output_add_readvariableop_resource*
_output_shapes
:	*
dtype0�
Pmodel_5/transformer_encoder_17/Encoder-SelfAttentionLayer-3/attention_output/addAddV2cmodel_5/transformer_encoder_17/Encoder-SelfAttentionLayer-3/attention_output/einsum/Einsum:output:0gmodel_5/transformer_encoder_17/Encoder-SelfAttentionLayer-3/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������P	�
>model_5/transformer_encoder_17/Encoder-1st-AdditionLayer-3/addAddV2Bmodel_5/transformer_encoder_16/Encoder-2nd-AdditionLayer-2/add:z:0Tmodel_5/transformer_encoder_17/Encoder-SelfAttentionLayer-3/attention_output/add:z:0*
T0*+
_output_shapes
:���������P	�
Emodel_5/transformer_encoder_17/Encoder-2nd-NormalizationLayer-3/ShapeShapeBmodel_5/transformer_encoder_17/Encoder-1st-AdditionLayer-3/add:z:0*
T0*
_output_shapes
::���
Smodel_5/transformer_encoder_17/Encoder-2nd-NormalizationLayer-3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
Umodel_5/transformer_encoder_17/Encoder-2nd-NormalizationLayer-3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
Umodel_5/transformer_encoder_17/Encoder-2nd-NormalizationLayer-3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
Mmodel_5/transformer_encoder_17/Encoder-2nd-NormalizationLayer-3/strided_sliceStridedSliceNmodel_5/transformer_encoder_17/Encoder-2nd-NormalizationLayer-3/Shape:output:0\model_5/transformer_encoder_17/Encoder-2nd-NormalizationLayer-3/strided_slice/stack:output:0^model_5/transformer_encoder_17/Encoder-2nd-NormalizationLayer-3/strided_slice/stack_1:output:0^model_5/transformer_encoder_17/Encoder-2nd-NormalizationLayer-3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask�
Emodel_5/transformer_encoder_17/Encoder-2nd-NormalizationLayer-3/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
Dmodel_5/transformer_encoder_17/Encoder-2nd-NormalizationLayer-3/ProdProdVmodel_5/transformer_encoder_17/Encoder-2nd-NormalizationLayer-3/strided_slice:output:0Nmodel_5/transformer_encoder_17/Encoder-2nd-NormalizationLayer-3/Const:output:0*
T0*
_output_shapes
: �
Umodel_5/transformer_encoder_17/Encoder-2nd-NormalizationLayer-3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:�
Wmodel_5/transformer_encoder_17/Encoder-2nd-NormalizationLayer-3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: �
Wmodel_5/transformer_encoder_17/Encoder-2nd-NormalizationLayer-3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
Omodel_5/transformer_encoder_17/Encoder-2nd-NormalizationLayer-3/strided_slice_1StridedSliceNmodel_5/transformer_encoder_17/Encoder-2nd-NormalizationLayer-3/Shape:output:0^model_5/transformer_encoder_17/Encoder-2nd-NormalizationLayer-3/strided_slice_1/stack:output:0`model_5/transformer_encoder_17/Encoder-2nd-NormalizationLayer-3/strided_slice_1/stack_1:output:0`model_5/transformer_encoder_17/Encoder-2nd-NormalizationLayer-3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask�
Gmodel_5/transformer_encoder_17/Encoder-2nd-NormalizationLayer-3/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
Fmodel_5/transformer_encoder_17/Encoder-2nd-NormalizationLayer-3/Prod_1ProdXmodel_5/transformer_encoder_17/Encoder-2nd-NormalizationLayer-3/strided_slice_1:output:0Pmodel_5/transformer_encoder_17/Encoder-2nd-NormalizationLayer-3/Const_1:output:0*
T0*
_output_shapes
: �
Omodel_5/transformer_encoder_17/Encoder-2nd-NormalizationLayer-3/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :�
Omodel_5/transformer_encoder_17/Encoder-2nd-NormalizationLayer-3/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :�
Mmodel_5/transformer_encoder_17/Encoder-2nd-NormalizationLayer-3/Reshape/shapePackXmodel_5/transformer_encoder_17/Encoder-2nd-NormalizationLayer-3/Reshape/shape/0:output:0Mmodel_5/transformer_encoder_17/Encoder-2nd-NormalizationLayer-3/Prod:output:0Omodel_5/transformer_encoder_17/Encoder-2nd-NormalizationLayer-3/Prod_1:output:0Xmodel_5/transformer_encoder_17/Encoder-2nd-NormalizationLayer-3/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:�
Gmodel_5/transformer_encoder_17/Encoder-2nd-NormalizationLayer-3/ReshapeReshapeBmodel_5/transformer_encoder_17/Encoder-1st-AdditionLayer-3/add:z:0Vmodel_5/transformer_encoder_17/Encoder-2nd-NormalizationLayer-3/Reshape/shape:output:0*
T0*8
_output_shapes&
$:"�������������������
Kmodel_5/transformer_encoder_17/Encoder-2nd-NormalizationLayer-3/ones/packedPackMmodel_5/transformer_encoder_17/Encoder-2nd-NormalizationLayer-3/Prod:output:0*
N*
T0*
_output_shapes
:�
Jmodel_5/transformer_encoder_17/Encoder-2nd-NormalizationLayer-3/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
Dmodel_5/transformer_encoder_17/Encoder-2nd-NormalizationLayer-3/onesFillTmodel_5/transformer_encoder_17/Encoder-2nd-NormalizationLayer-3/ones/packed:output:0Smodel_5/transformer_encoder_17/Encoder-2nd-NormalizationLayer-3/ones/Const:output:0*
T0*#
_output_shapes
:����������
Lmodel_5/transformer_encoder_17/Encoder-2nd-NormalizationLayer-3/zeros/packedPackMmodel_5/transformer_encoder_17/Encoder-2nd-NormalizationLayer-3/Prod:output:0*
N*
T0*
_output_shapes
:�
Kmodel_5/transformer_encoder_17/Encoder-2nd-NormalizationLayer-3/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
Emodel_5/transformer_encoder_17/Encoder-2nd-NormalizationLayer-3/zerosFillUmodel_5/transformer_encoder_17/Encoder-2nd-NormalizationLayer-3/zeros/packed:output:0Tmodel_5/transformer_encoder_17/Encoder-2nd-NormalizationLayer-3/zeros/Const:output:0*
T0*#
_output_shapes
:����������
Gmodel_5/transformer_encoder_17/Encoder-2nd-NormalizationLayer-3/Const_2Const*
_output_shapes
: *
dtype0*
valueB �
Gmodel_5/transformer_encoder_17/Encoder-2nd-NormalizationLayer-3/Const_3Const*
_output_shapes
: *
dtype0*
valueB �
Pmodel_5/transformer_encoder_17/Encoder-2nd-NormalizationLayer-3/FusedBatchNormV3FusedBatchNormV3Pmodel_5/transformer_encoder_17/Encoder-2nd-NormalizationLayer-3/Reshape:output:0Mmodel_5/transformer_encoder_17/Encoder-2nd-NormalizationLayer-3/ones:output:0Nmodel_5/transformer_encoder_17/Encoder-2nd-NormalizationLayer-3/zeros:output:0Pmodel_5/transformer_encoder_17/Encoder-2nd-NormalizationLayer-3/Const_2:output:0Pmodel_5/transformer_encoder_17/Encoder-2nd-NormalizationLayer-3/Const_3:output:0*
T0*
U0*x
_output_shapesf
d:"������������������:���������:���������:���������:���������:*
data_formatNCHW*
epsilon%o�:�
Imodel_5/transformer_encoder_17/Encoder-2nd-NormalizationLayer-3/Reshape_1ReshapeTmodel_5/transformer_encoder_17/Encoder-2nd-NormalizationLayer-3/FusedBatchNormV3:y:0Nmodel_5/transformer_encoder_17/Encoder-2nd-NormalizationLayer-3/Shape:output:0*
T0*+
_output_shapes
:���������P	�
Rmodel_5/transformer_encoder_17/Encoder-2nd-NormalizationLayer-3/mul/ReadVariableOpReadVariableOp[model_5_transformer_encoder_17_encoder_2nd_normalizationlayer_3_mul_readvariableop_resource*
_output_shapes
:	*
dtype0�
Cmodel_5/transformer_encoder_17/Encoder-2nd-NormalizationLayer-3/mulMulRmodel_5/transformer_encoder_17/Encoder-2nd-NormalizationLayer-3/Reshape_1:output:0Zmodel_5/transformer_encoder_17/Encoder-2nd-NormalizationLayer-3/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������P	�
Rmodel_5/transformer_encoder_17/Encoder-2nd-NormalizationLayer-3/add/ReadVariableOpReadVariableOp[model_5_transformer_encoder_17_encoder_2nd_normalizationlayer_3_add_readvariableop_resource*
_output_shapes
:	*
dtype0�
Cmodel_5/transformer_encoder_17/Encoder-2nd-NormalizationLayer-3/addAddV2Gmodel_5/transformer_encoder_17/Encoder-2nd-NormalizationLayer-3/mul:z:0Zmodel_5/transformer_encoder_17/Encoder-2nd-NormalizationLayer-3/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������P	�
Tmodel_5/transformer_encoder_17/Encoder-FeedForwardLayer_1_3/Tensordot/ReadVariableOpReadVariableOp]model_5_transformer_encoder_17_encoder_feedforwardlayer_1_3_tensordot_readvariableop_resource*
_output_shapes

:		*
dtype0�
Jmodel_5/transformer_encoder_17/Encoder-FeedForwardLayer_1_3/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:�
Jmodel_5/transformer_encoder_17/Encoder-FeedForwardLayer_1_3/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       �
Kmodel_5/transformer_encoder_17/Encoder-FeedForwardLayer_1_3/Tensordot/ShapeShapeGmodel_5/transformer_encoder_17/Encoder-2nd-NormalizationLayer-3/add:z:0*
T0*
_output_shapes
::���
Smodel_5/transformer_encoder_17/Encoder-FeedForwardLayer_1_3/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Nmodel_5/transformer_encoder_17/Encoder-FeedForwardLayer_1_3/Tensordot/GatherV2GatherV2Tmodel_5/transformer_encoder_17/Encoder-FeedForwardLayer_1_3/Tensordot/Shape:output:0Smodel_5/transformer_encoder_17/Encoder-FeedForwardLayer_1_3/Tensordot/free:output:0\model_5/transformer_encoder_17/Encoder-FeedForwardLayer_1_3/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
Umodel_5/transformer_encoder_17/Encoder-FeedForwardLayer_1_3/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Pmodel_5/transformer_encoder_17/Encoder-FeedForwardLayer_1_3/Tensordot/GatherV2_1GatherV2Tmodel_5/transformer_encoder_17/Encoder-FeedForwardLayer_1_3/Tensordot/Shape:output:0Smodel_5/transformer_encoder_17/Encoder-FeedForwardLayer_1_3/Tensordot/axes:output:0^model_5/transformer_encoder_17/Encoder-FeedForwardLayer_1_3/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
Kmodel_5/transformer_encoder_17/Encoder-FeedForwardLayer_1_3/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
Jmodel_5/transformer_encoder_17/Encoder-FeedForwardLayer_1_3/Tensordot/ProdProdWmodel_5/transformer_encoder_17/Encoder-FeedForwardLayer_1_3/Tensordot/GatherV2:output:0Tmodel_5/transformer_encoder_17/Encoder-FeedForwardLayer_1_3/Tensordot/Const:output:0*
T0*
_output_shapes
: �
Mmodel_5/transformer_encoder_17/Encoder-FeedForwardLayer_1_3/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
Lmodel_5/transformer_encoder_17/Encoder-FeedForwardLayer_1_3/Tensordot/Prod_1ProdYmodel_5/transformer_encoder_17/Encoder-FeedForwardLayer_1_3/Tensordot/GatherV2_1:output:0Vmodel_5/transformer_encoder_17/Encoder-FeedForwardLayer_1_3/Tensordot/Const_1:output:0*
T0*
_output_shapes
: �
Qmodel_5/transformer_encoder_17/Encoder-FeedForwardLayer_1_3/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Lmodel_5/transformer_encoder_17/Encoder-FeedForwardLayer_1_3/Tensordot/concatConcatV2Smodel_5/transformer_encoder_17/Encoder-FeedForwardLayer_1_3/Tensordot/free:output:0Smodel_5/transformer_encoder_17/Encoder-FeedForwardLayer_1_3/Tensordot/axes:output:0Zmodel_5/transformer_encoder_17/Encoder-FeedForwardLayer_1_3/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
Kmodel_5/transformer_encoder_17/Encoder-FeedForwardLayer_1_3/Tensordot/stackPackSmodel_5/transformer_encoder_17/Encoder-FeedForwardLayer_1_3/Tensordot/Prod:output:0Umodel_5/transformer_encoder_17/Encoder-FeedForwardLayer_1_3/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
Omodel_5/transformer_encoder_17/Encoder-FeedForwardLayer_1_3/Tensordot/transpose	TransposeGmodel_5/transformer_encoder_17/Encoder-2nd-NormalizationLayer-3/add:z:0Umodel_5/transformer_encoder_17/Encoder-FeedForwardLayer_1_3/Tensordot/concat:output:0*
T0*+
_output_shapes
:���������P	�
Mmodel_5/transformer_encoder_17/Encoder-FeedForwardLayer_1_3/Tensordot/ReshapeReshapeSmodel_5/transformer_encoder_17/Encoder-FeedForwardLayer_1_3/Tensordot/transpose:y:0Tmodel_5/transformer_encoder_17/Encoder-FeedForwardLayer_1_3/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
Lmodel_5/transformer_encoder_17/Encoder-FeedForwardLayer_1_3/Tensordot/MatMulMatMulVmodel_5/transformer_encoder_17/Encoder-FeedForwardLayer_1_3/Tensordot/Reshape:output:0\model_5/transformer_encoder_17/Encoder-FeedForwardLayer_1_3/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������	�
Mmodel_5/transformer_encoder_17/Encoder-FeedForwardLayer_1_3/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:	�
Smodel_5/transformer_encoder_17/Encoder-FeedForwardLayer_1_3/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Nmodel_5/transformer_encoder_17/Encoder-FeedForwardLayer_1_3/Tensordot/concat_1ConcatV2Wmodel_5/transformer_encoder_17/Encoder-FeedForwardLayer_1_3/Tensordot/GatherV2:output:0Vmodel_5/transformer_encoder_17/Encoder-FeedForwardLayer_1_3/Tensordot/Const_2:output:0\model_5/transformer_encoder_17/Encoder-FeedForwardLayer_1_3/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
Emodel_5/transformer_encoder_17/Encoder-FeedForwardLayer_1_3/TensordotReshapeVmodel_5/transformer_encoder_17/Encoder-FeedForwardLayer_1_3/Tensordot/MatMul:product:0Wmodel_5/transformer_encoder_17/Encoder-FeedForwardLayer_1_3/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������P	�
Rmodel_5/transformer_encoder_17/Encoder-FeedForwardLayer_1_3/BiasAdd/ReadVariableOpReadVariableOp[model_5_transformer_encoder_17_encoder_feedforwardlayer_1_3_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype0�
Cmodel_5/transformer_encoder_17/Encoder-FeedForwardLayer_1_3/BiasAddBiasAddNmodel_5/transformer_encoder_17/Encoder-FeedForwardLayer_1_3/Tensordot:output:0Zmodel_5/transformer_encoder_17/Encoder-FeedForwardLayer_1_3/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������P	�
Fmodel_5/transformer_encoder_17/Encoder-FeedForwardLayer_1_3/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
Dmodel_5/transformer_encoder_17/Encoder-FeedForwardLayer_1_3/Gelu/mulMulOmodel_5/transformer_encoder_17/Encoder-FeedForwardLayer_1_3/Gelu/mul/x:output:0Lmodel_5/transformer_encoder_17/Encoder-FeedForwardLayer_1_3/BiasAdd:output:0*
T0*+
_output_shapes
:���������P	�
Gmodel_5/transformer_encoder_17/Encoder-FeedForwardLayer_1_3/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *��?�
Hmodel_5/transformer_encoder_17/Encoder-FeedForwardLayer_1_3/Gelu/truedivRealDivLmodel_5/transformer_encoder_17/Encoder-FeedForwardLayer_1_3/BiasAdd:output:0Pmodel_5/transformer_encoder_17/Encoder-FeedForwardLayer_1_3/Gelu/Cast/x:output:0*
T0*+
_output_shapes
:���������P	�
Dmodel_5/transformer_encoder_17/Encoder-FeedForwardLayer_1_3/Gelu/ErfErfLmodel_5/transformer_encoder_17/Encoder-FeedForwardLayer_1_3/Gelu/truediv:z:0*
T0*+
_output_shapes
:���������P	�
Fmodel_5/transformer_encoder_17/Encoder-FeedForwardLayer_1_3/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
Dmodel_5/transformer_encoder_17/Encoder-FeedForwardLayer_1_3/Gelu/addAddV2Omodel_5/transformer_encoder_17/Encoder-FeedForwardLayer_1_3/Gelu/add/x:output:0Hmodel_5/transformer_encoder_17/Encoder-FeedForwardLayer_1_3/Gelu/Erf:y:0*
T0*+
_output_shapes
:���������P	�
Fmodel_5/transformer_encoder_17/Encoder-FeedForwardLayer_1_3/Gelu/mul_1MulHmodel_5/transformer_encoder_17/Encoder-FeedForwardLayer_1_3/Gelu/mul:z:0Hmodel_5/transformer_encoder_17/Encoder-FeedForwardLayer_1_3/Gelu/add:z:0*
T0*+
_output_shapes
:���������P	�
Tmodel_5/transformer_encoder_17/Encoder-FeedForwardLayer_2_3/Tensordot/ReadVariableOpReadVariableOp]model_5_transformer_encoder_17_encoder_feedforwardlayer_2_3_tensordot_readvariableop_resource*
_output_shapes

:		*
dtype0�
Jmodel_5/transformer_encoder_17/Encoder-FeedForwardLayer_2_3/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:�
Jmodel_5/transformer_encoder_17/Encoder-FeedForwardLayer_2_3/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       �
Kmodel_5/transformer_encoder_17/Encoder-FeedForwardLayer_2_3/Tensordot/ShapeShapeJmodel_5/transformer_encoder_17/Encoder-FeedForwardLayer_1_3/Gelu/mul_1:z:0*
T0*
_output_shapes
::���
Smodel_5/transformer_encoder_17/Encoder-FeedForwardLayer_2_3/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Nmodel_5/transformer_encoder_17/Encoder-FeedForwardLayer_2_3/Tensordot/GatherV2GatherV2Tmodel_5/transformer_encoder_17/Encoder-FeedForwardLayer_2_3/Tensordot/Shape:output:0Smodel_5/transformer_encoder_17/Encoder-FeedForwardLayer_2_3/Tensordot/free:output:0\model_5/transformer_encoder_17/Encoder-FeedForwardLayer_2_3/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
Umodel_5/transformer_encoder_17/Encoder-FeedForwardLayer_2_3/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Pmodel_5/transformer_encoder_17/Encoder-FeedForwardLayer_2_3/Tensordot/GatherV2_1GatherV2Tmodel_5/transformer_encoder_17/Encoder-FeedForwardLayer_2_3/Tensordot/Shape:output:0Smodel_5/transformer_encoder_17/Encoder-FeedForwardLayer_2_3/Tensordot/axes:output:0^model_5/transformer_encoder_17/Encoder-FeedForwardLayer_2_3/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
Kmodel_5/transformer_encoder_17/Encoder-FeedForwardLayer_2_3/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
Jmodel_5/transformer_encoder_17/Encoder-FeedForwardLayer_2_3/Tensordot/ProdProdWmodel_5/transformer_encoder_17/Encoder-FeedForwardLayer_2_3/Tensordot/GatherV2:output:0Tmodel_5/transformer_encoder_17/Encoder-FeedForwardLayer_2_3/Tensordot/Const:output:0*
T0*
_output_shapes
: �
Mmodel_5/transformer_encoder_17/Encoder-FeedForwardLayer_2_3/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
Lmodel_5/transformer_encoder_17/Encoder-FeedForwardLayer_2_3/Tensordot/Prod_1ProdYmodel_5/transformer_encoder_17/Encoder-FeedForwardLayer_2_3/Tensordot/GatherV2_1:output:0Vmodel_5/transformer_encoder_17/Encoder-FeedForwardLayer_2_3/Tensordot/Const_1:output:0*
T0*
_output_shapes
: �
Qmodel_5/transformer_encoder_17/Encoder-FeedForwardLayer_2_3/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Lmodel_5/transformer_encoder_17/Encoder-FeedForwardLayer_2_3/Tensordot/concatConcatV2Smodel_5/transformer_encoder_17/Encoder-FeedForwardLayer_2_3/Tensordot/free:output:0Smodel_5/transformer_encoder_17/Encoder-FeedForwardLayer_2_3/Tensordot/axes:output:0Zmodel_5/transformer_encoder_17/Encoder-FeedForwardLayer_2_3/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
Kmodel_5/transformer_encoder_17/Encoder-FeedForwardLayer_2_3/Tensordot/stackPackSmodel_5/transformer_encoder_17/Encoder-FeedForwardLayer_2_3/Tensordot/Prod:output:0Umodel_5/transformer_encoder_17/Encoder-FeedForwardLayer_2_3/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
Omodel_5/transformer_encoder_17/Encoder-FeedForwardLayer_2_3/Tensordot/transpose	TransposeJmodel_5/transformer_encoder_17/Encoder-FeedForwardLayer_1_3/Gelu/mul_1:z:0Umodel_5/transformer_encoder_17/Encoder-FeedForwardLayer_2_3/Tensordot/concat:output:0*
T0*+
_output_shapes
:���������P	�
Mmodel_5/transformer_encoder_17/Encoder-FeedForwardLayer_2_3/Tensordot/ReshapeReshapeSmodel_5/transformer_encoder_17/Encoder-FeedForwardLayer_2_3/Tensordot/transpose:y:0Tmodel_5/transformer_encoder_17/Encoder-FeedForwardLayer_2_3/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
Lmodel_5/transformer_encoder_17/Encoder-FeedForwardLayer_2_3/Tensordot/MatMulMatMulVmodel_5/transformer_encoder_17/Encoder-FeedForwardLayer_2_3/Tensordot/Reshape:output:0\model_5/transformer_encoder_17/Encoder-FeedForwardLayer_2_3/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������	�
Mmodel_5/transformer_encoder_17/Encoder-FeedForwardLayer_2_3/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:	�
Smodel_5/transformer_encoder_17/Encoder-FeedForwardLayer_2_3/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Nmodel_5/transformer_encoder_17/Encoder-FeedForwardLayer_2_3/Tensordot/concat_1ConcatV2Wmodel_5/transformer_encoder_17/Encoder-FeedForwardLayer_2_3/Tensordot/GatherV2:output:0Vmodel_5/transformer_encoder_17/Encoder-FeedForwardLayer_2_3/Tensordot/Const_2:output:0\model_5/transformer_encoder_17/Encoder-FeedForwardLayer_2_3/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
Emodel_5/transformer_encoder_17/Encoder-FeedForwardLayer_2_3/TensordotReshapeVmodel_5/transformer_encoder_17/Encoder-FeedForwardLayer_2_3/Tensordot/MatMul:product:0Wmodel_5/transformer_encoder_17/Encoder-FeedForwardLayer_2_3/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������P	�
Rmodel_5/transformer_encoder_17/Encoder-FeedForwardLayer_2_3/BiasAdd/ReadVariableOpReadVariableOp[model_5_transformer_encoder_17_encoder_feedforwardlayer_2_3_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype0�
Cmodel_5/transformer_encoder_17/Encoder-FeedForwardLayer_2_3/BiasAddBiasAddNmodel_5/transformer_encoder_17/Encoder-FeedForwardLayer_2_3/Tensordot:output:0Zmodel_5/transformer_encoder_17/Encoder-FeedForwardLayer_2_3/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������P	�
2model_5/transformer_encoder_17/dropout_17/IdentityIdentityLmodel_5/transformer_encoder_17/Encoder-FeedForwardLayer_2_3/BiasAdd:output:0*
T0*+
_output_shapes
:���������P	�
>model_5/transformer_encoder_17/Encoder-2nd-AdditionLayer-3/addAddV2Bmodel_5/transformer_encoder_17/Encoder-1st-AdditionLayer-3/add:z:0;model_5/transformer_encoder_17/dropout_17/Identity:output:0*
T0*+
_output_shapes
:���������P	�
model_5/FinalLayerNorm/ShapeShapeBmodel_5/transformer_encoder_17/Encoder-2nd-AdditionLayer-3/add:z:0*
T0*
_output_shapes
::��t
*model_5/FinalLayerNorm/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: v
,model_5/FinalLayerNorm/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:v
,model_5/FinalLayerNorm/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
$model_5/FinalLayerNorm/strided_sliceStridedSlice%model_5/FinalLayerNorm/Shape:output:03model_5/FinalLayerNorm/strided_slice/stack:output:05model_5/FinalLayerNorm/strided_slice/stack_1:output:05model_5/FinalLayerNorm/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskf
model_5/FinalLayerNorm/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
model_5/FinalLayerNorm/ProdProd-model_5/FinalLayerNorm/strided_slice:output:0%model_5/FinalLayerNorm/Const:output:0*
T0*
_output_shapes
: v
,model_5/FinalLayerNorm/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:x
.model_5/FinalLayerNorm/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: x
.model_5/FinalLayerNorm/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
&model_5/FinalLayerNorm/strided_slice_1StridedSlice%model_5/FinalLayerNorm/Shape:output:05model_5/FinalLayerNorm/strided_slice_1/stack:output:07model_5/FinalLayerNorm/strided_slice_1/stack_1:output:07model_5/FinalLayerNorm/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_maskh
model_5/FinalLayerNorm/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
model_5/FinalLayerNorm/Prod_1Prod/model_5/FinalLayerNorm/strided_slice_1:output:0'model_5/FinalLayerNorm/Const_1:output:0*
T0*
_output_shapes
: h
&model_5/FinalLayerNorm/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :h
&model_5/FinalLayerNorm/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :�
$model_5/FinalLayerNorm/Reshape/shapePack/model_5/FinalLayerNorm/Reshape/shape/0:output:0$model_5/FinalLayerNorm/Prod:output:0&model_5/FinalLayerNorm/Prod_1:output:0/model_5/FinalLayerNorm/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:�
model_5/FinalLayerNorm/ReshapeReshapeBmodel_5/transformer_encoder_17/Encoder-2nd-AdditionLayer-3/add:z:0-model_5/FinalLayerNorm/Reshape/shape:output:0*
T0*8
_output_shapes&
$:"������������������~
"model_5/FinalLayerNorm/ones/packedPack$model_5/FinalLayerNorm/Prod:output:0*
N*
T0*
_output_shapes
:f
!model_5/FinalLayerNorm/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
model_5/FinalLayerNorm/onesFill+model_5/FinalLayerNorm/ones/packed:output:0*model_5/FinalLayerNorm/ones/Const:output:0*
T0*#
_output_shapes
:���������
#model_5/FinalLayerNorm/zeros/packedPack$model_5/FinalLayerNorm/Prod:output:0*
N*
T0*
_output_shapes
:g
"model_5/FinalLayerNorm/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
model_5/FinalLayerNorm/zerosFill,model_5/FinalLayerNorm/zeros/packed:output:0+model_5/FinalLayerNorm/zeros/Const:output:0*
T0*#
_output_shapes
:���������a
model_5/FinalLayerNorm/Const_2Const*
_output_shapes
: *
dtype0*
valueB a
model_5/FinalLayerNorm/Const_3Const*
_output_shapes
: *
dtype0*
valueB �
'model_5/FinalLayerNorm/FusedBatchNormV3FusedBatchNormV3'model_5/FinalLayerNorm/Reshape:output:0$model_5/FinalLayerNorm/ones:output:0%model_5/FinalLayerNorm/zeros:output:0'model_5/FinalLayerNorm/Const_2:output:0'model_5/FinalLayerNorm/Const_3:output:0*
T0*
U0*x
_output_shapesf
d:"������������������:���������:���������:���������:���������:*
data_formatNCHW*
epsilon%o�:�
 model_5/FinalLayerNorm/Reshape_1Reshape+model_5/FinalLayerNorm/FusedBatchNormV3:y:0%model_5/FinalLayerNorm/Shape:output:0*
T0*+
_output_shapes
:���������P	�
)model_5/FinalLayerNorm/mul/ReadVariableOpReadVariableOp2model_5_finallayernorm_mul_readvariableop_resource*
_output_shapes
:	*
dtype0�
model_5/FinalLayerNorm/mulMul)model_5/FinalLayerNorm/Reshape_1:output:01model_5/FinalLayerNorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������P	�
)model_5/FinalLayerNorm/add/ReadVariableOpReadVariableOp2model_5_finallayernorm_add_readvariableop_resource*
_output_shapes
:	*
dtype0�
model_5/FinalLayerNorm/addAddV2model_5/FinalLayerNorm/mul:z:01model_5/FinalLayerNorm/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������P	�
>model_5/ReduceStackDimensionViaSummation/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :�
,model_5/ReduceStackDimensionViaSummation/SumSummodel_5/FinalLayerNorm/add:z:0Gmodel_5/ReduceStackDimensionViaSummation/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:���������	w
2model_5/ReduceStackDimensionViaSummation/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �B�
0model_5/ReduceStackDimensionViaSummation/truedivRealDiv5model_5/ReduceStackDimensionViaSummation/Sum:output:0;model_5/ReduceStackDimensionViaSummation/truediv/y:output:0*
T0*'
_output_shapes
:���������	z
!model_5/StandardizeTimeLimit/CastCasttimelimitinput*

DstT0*

SrcT0*'
_output_shapes
:���������g
"model_5/StandardizeTimeLimit/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  pA�
 model_5/StandardizeTimeLimit/subSub%model_5/StandardizeTimeLimit/Cast:y:0+model_5/StandardizeTimeLimit/sub/y:output:0*
T0*'
_output_shapes
:���������k
&model_5/StandardizeTimeLimit/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �@�
$model_5/StandardizeTimeLimit/truedivRealDiv$model_5/StandardizeTimeLimit/sub:z:0/model_5/StandardizeTimeLimit/truediv/y:output:0*
T0*'
_output_shapes
:���������f
$model_5/ConcatenateLayer/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
model_5/ConcatenateLayer/concatConcatV24model_5/ReduceStackDimensionViaSummation/truediv:z:0(model_5/StandardizeTimeLimit/truediv:z:0-model_5/ConcatenateLayer/concat/axis:output:0*
N*
T0*'
_output_shapes
:���������
h
&model_5/ConcatenateLayer/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :�
!model_5/ConcatenateLayer/concat_1ConcatV24model_5/ReduceStackDimensionViaSummation/truediv:z:0(model_5/StandardizeTimeLimit/truediv:z:0/model_5/ConcatenateLayer/concat_1/axis:output:0*
N*
T0*'
_output_shapes
:���������
�
7model_5/FullyConnectedLayerSolved/MatMul/ReadVariableOpReadVariableOp@model_5_fullyconnectedlayersolved_matmul_readvariableop_resource*
_output_shapes

:

*
dtype0�
(model_5/FullyConnectedLayerSolved/MatMulMatMul(model_5/ConcatenateLayer/concat:output:0?model_5/FullyConnectedLayerSolved/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
�
8model_5/FullyConnectedLayerSolved/BiasAdd/ReadVariableOpReadVariableOpAmodel_5_fullyconnectedlayersolved_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0�
)model_5/FullyConnectedLayerSolved/BiasAddBiasAdd2model_5/FullyConnectedLayerSolved/MatMul:product:0@model_5/FullyConnectedLayerSolved/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
q
,model_5/FullyConnectedLayerSolved/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
*model_5/FullyConnectedLayerSolved/Gelu/mulMul5model_5/FullyConnectedLayerSolved/Gelu/mul/x:output:02model_5/FullyConnectedLayerSolved/BiasAdd:output:0*
T0*'
_output_shapes
:���������
r
-model_5/FullyConnectedLayerSolved/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *��?�
.model_5/FullyConnectedLayerSolved/Gelu/truedivRealDiv2model_5/FullyConnectedLayerSolved/BiasAdd:output:06model_5/FullyConnectedLayerSolved/Gelu/Cast/x:output:0*
T0*'
_output_shapes
:���������
�
*model_5/FullyConnectedLayerSolved/Gelu/ErfErf2model_5/FullyConnectedLayerSolved/Gelu/truediv:z:0*
T0*'
_output_shapes
:���������
q
,model_5/FullyConnectedLayerSolved/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
*model_5/FullyConnectedLayerSolved/Gelu/addAddV25model_5/FullyConnectedLayerSolved/Gelu/add/x:output:0.model_5/FullyConnectedLayerSolved/Gelu/Erf:y:0*
T0*'
_output_shapes
:���������
�
,model_5/FullyConnectedLayerSolved/Gelu/mul_1Mul.model_5/FullyConnectedLayerSolved/Gelu/mul:z:0.model_5/FullyConnectedLayerSolved/Gelu/add:z:0*
T0*'
_output_shapes
:���������
�
:model_5/FullyConnectedLayerAreaRatio/MatMul/ReadVariableOpReadVariableOpCmodel_5_fullyconnectedlayerarearatio_matmul_readvariableop_resource*
_output_shapes

:

*
dtype0�
+model_5/FullyConnectedLayerAreaRatio/MatMulMatMul*model_5/ConcatenateLayer/concat_1:output:0Bmodel_5/FullyConnectedLayerAreaRatio/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
�
;model_5/FullyConnectedLayerAreaRatio/BiasAdd/ReadVariableOpReadVariableOpDmodel_5_fullyconnectedlayerarearatio_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0�
,model_5/FullyConnectedLayerAreaRatio/BiasAddBiasAdd5model_5/FullyConnectedLayerAreaRatio/MatMul:product:0Cmodel_5/FullyConnectedLayerAreaRatio/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
t
/model_5/FullyConnectedLayerAreaRatio/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
-model_5/FullyConnectedLayerAreaRatio/Gelu/mulMul8model_5/FullyConnectedLayerAreaRatio/Gelu/mul/x:output:05model_5/FullyConnectedLayerAreaRatio/BiasAdd:output:0*
T0*'
_output_shapes
:���������
u
0model_5/FullyConnectedLayerAreaRatio/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *��?�
1model_5/FullyConnectedLayerAreaRatio/Gelu/truedivRealDiv5model_5/FullyConnectedLayerAreaRatio/BiasAdd:output:09model_5/FullyConnectedLayerAreaRatio/Gelu/Cast/x:output:0*
T0*'
_output_shapes
:���������
�
-model_5/FullyConnectedLayerAreaRatio/Gelu/ErfErf5model_5/FullyConnectedLayerAreaRatio/Gelu/truediv:z:0*
T0*'
_output_shapes
:���������
t
/model_5/FullyConnectedLayerAreaRatio/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
-model_5/FullyConnectedLayerAreaRatio/Gelu/addAddV28model_5/FullyConnectedLayerAreaRatio/Gelu/add/x:output:01model_5/FullyConnectedLayerAreaRatio/Gelu/Erf:y:0*
T0*'
_output_shapes
:���������
�
/model_5/FullyConnectedLayerAreaRatio/Gelu/mul_1Mul1model_5/FullyConnectedLayerAreaRatio/Gelu/mul:z:01model_5/FullyConnectedLayerAreaRatio/Gelu/add:z:0*
T0*'
_output_shapes
:���������
�
.model_5/PredictionSolved/MatMul/ReadVariableOpReadVariableOp7model_5_predictionsolved_matmul_readvariableop_resource*
_output_shapes

:
*
dtype0�
model_5/PredictionSolved/MatMulMatMul0model_5/FullyConnectedLayerSolved/Gelu/mul_1:z:06model_5/PredictionSolved/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
/model_5/PredictionSolved/BiasAdd/ReadVariableOpReadVariableOp8model_5_predictionsolved_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
 model_5/PredictionSolved/BiasAddBiasAdd)model_5/PredictionSolved/MatMul:product:07model_5/PredictionSolved/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
 model_5/PredictionSolved/SigmoidSigmoid)model_5/PredictionSolved/BiasAdd:output:0*
T0*'
_output_shapes
:����������
1model_5/PredictionAreaRatio/MatMul/ReadVariableOpReadVariableOp:model_5_predictionarearatio_matmul_readvariableop_resource*
_output_shapes

:
*
dtype0�
"model_5/PredictionAreaRatio/MatMulMatMul3model_5/FullyConnectedLayerAreaRatio/Gelu/mul_1:z:09model_5/PredictionAreaRatio/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
2model_5/PredictionAreaRatio/BiasAdd/ReadVariableOpReadVariableOp;model_5_predictionarearatio_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
#model_5/PredictionAreaRatio/BiasAddBiasAdd,model_5/PredictionAreaRatio/MatMul:product:0:model_5/PredictionAreaRatio/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
#model_5/PredictionAreaRatio/SigmoidSigmoid,model_5/PredictionAreaRatio/BiasAdd:output:0*
T0*'
_output_shapes
:���������j
model_5/Output/SqueezeSqueeze$model_5/PredictionSolved/Sigmoid:y:0*
T0*
_output_shapes
:o
model_5/Output/Squeeze_1Squeeze'model_5/PredictionAreaRatio/Sigmoid:y:0*
T0*
_output_shapes
:a
IdentityIdentity!model_5/Output/Squeeze_1:output:0^NoOp*
T0*
_output_shapes
:a

Identity_1Identitymodel_5/Output/Squeeze:output:0^NoOp*
T0*
_output_shapes
:�&
NoOpNoOp*^model_5/FinalLayerNorm/add/ReadVariableOp*^model_5/FinalLayerNorm/mul/ReadVariableOp<^model_5/FullyConnectedLayerAreaRatio/BiasAdd/ReadVariableOp;^model_5/FullyConnectedLayerAreaRatio/MatMul/ReadVariableOp9^model_5/FullyConnectedLayerSolved/BiasAdd/ReadVariableOp8^model_5/FullyConnectedLayerSolved/MatMul/ReadVariableOp3^model_5/PredictionAreaRatio/BiasAdd/ReadVariableOp2^model_5/PredictionAreaRatio/MatMul/ReadVariableOp0^model_5/PredictionSolved/BiasAdd/ReadVariableOp/^model_5/PredictionSolved/MatMul/ReadVariableOpS^model_5/transformer_encoder_15/Encoder-1st-NormalizationLayer-1/add/ReadVariableOpS^model_5/transformer_encoder_15/Encoder-1st-NormalizationLayer-1/mul/ReadVariableOpS^model_5/transformer_encoder_15/Encoder-2nd-NormalizationLayer-1/add/ReadVariableOpS^model_5/transformer_encoder_15/Encoder-2nd-NormalizationLayer-1/mul/ReadVariableOpS^model_5/transformer_encoder_15/Encoder-FeedForwardLayer_1_1/BiasAdd/ReadVariableOpU^model_5/transformer_encoder_15/Encoder-FeedForwardLayer_1_1/Tensordot/ReadVariableOpS^model_5/transformer_encoder_15/Encoder-FeedForwardLayer_2_1/BiasAdd/ReadVariableOpU^model_5/transformer_encoder_15/Encoder-FeedForwardLayer_2_1/Tensordot/ReadVariableOp`^model_5/transformer_encoder_15/Encoder-SelfAttentionLayer-1/attention_output/add/ReadVariableOpj^model_5/transformer_encoder_15/Encoder-SelfAttentionLayer-1/attention_output/einsum/Einsum/ReadVariableOpS^model_5/transformer_encoder_15/Encoder-SelfAttentionLayer-1/key/add/ReadVariableOp]^model_5/transformer_encoder_15/Encoder-SelfAttentionLayer-1/key/einsum/Einsum/ReadVariableOpU^model_5/transformer_encoder_15/Encoder-SelfAttentionLayer-1/query/add/ReadVariableOp_^model_5/transformer_encoder_15/Encoder-SelfAttentionLayer-1/query/einsum/Einsum/ReadVariableOpU^model_5/transformer_encoder_15/Encoder-SelfAttentionLayer-1/value/add/ReadVariableOp_^model_5/transformer_encoder_15/Encoder-SelfAttentionLayer-1/value/einsum/Einsum/ReadVariableOpS^model_5/transformer_encoder_16/Encoder-1st-NormalizationLayer-2/add/ReadVariableOpS^model_5/transformer_encoder_16/Encoder-1st-NormalizationLayer-2/mul/ReadVariableOpS^model_5/transformer_encoder_16/Encoder-2nd-NormalizationLayer-2/add/ReadVariableOpS^model_5/transformer_encoder_16/Encoder-2nd-NormalizationLayer-2/mul/ReadVariableOpS^model_5/transformer_encoder_16/Encoder-FeedForwardLayer_1_2/BiasAdd/ReadVariableOpU^model_5/transformer_encoder_16/Encoder-FeedForwardLayer_1_2/Tensordot/ReadVariableOpS^model_5/transformer_encoder_16/Encoder-FeedForwardLayer_2_2/BiasAdd/ReadVariableOpU^model_5/transformer_encoder_16/Encoder-FeedForwardLayer_2_2/Tensordot/ReadVariableOp`^model_5/transformer_encoder_16/Encoder-SelfAttentionLayer-2/attention_output/add/ReadVariableOpj^model_5/transformer_encoder_16/Encoder-SelfAttentionLayer-2/attention_output/einsum/Einsum/ReadVariableOpS^model_5/transformer_encoder_16/Encoder-SelfAttentionLayer-2/key/add/ReadVariableOp]^model_5/transformer_encoder_16/Encoder-SelfAttentionLayer-2/key/einsum/Einsum/ReadVariableOpU^model_5/transformer_encoder_16/Encoder-SelfAttentionLayer-2/query/add/ReadVariableOp_^model_5/transformer_encoder_16/Encoder-SelfAttentionLayer-2/query/einsum/Einsum/ReadVariableOpU^model_5/transformer_encoder_16/Encoder-SelfAttentionLayer-2/value/add/ReadVariableOp_^model_5/transformer_encoder_16/Encoder-SelfAttentionLayer-2/value/einsum/Einsum/ReadVariableOpS^model_5/transformer_encoder_17/Encoder-1st-NormalizationLayer-3/add/ReadVariableOpS^model_5/transformer_encoder_17/Encoder-1st-NormalizationLayer-3/mul/ReadVariableOpS^model_5/transformer_encoder_17/Encoder-2nd-NormalizationLayer-3/add/ReadVariableOpS^model_5/transformer_encoder_17/Encoder-2nd-NormalizationLayer-3/mul/ReadVariableOpS^model_5/transformer_encoder_17/Encoder-FeedForwardLayer_1_3/BiasAdd/ReadVariableOpU^model_5/transformer_encoder_17/Encoder-FeedForwardLayer_1_3/Tensordot/ReadVariableOpS^model_5/transformer_encoder_17/Encoder-FeedForwardLayer_2_3/BiasAdd/ReadVariableOpU^model_5/transformer_encoder_17/Encoder-FeedForwardLayer_2_3/Tensordot/ReadVariableOp`^model_5/transformer_encoder_17/Encoder-SelfAttentionLayer-3/attention_output/add/ReadVariableOpj^model_5/transformer_encoder_17/Encoder-SelfAttentionLayer-3/attention_output/einsum/Einsum/ReadVariableOpS^model_5/transformer_encoder_17/Encoder-SelfAttentionLayer-3/key/add/ReadVariableOp]^model_5/transformer_encoder_17/Encoder-SelfAttentionLayer-3/key/einsum/Einsum/ReadVariableOpU^model_5/transformer_encoder_17/Encoder-SelfAttentionLayer-3/query/add/ReadVariableOp_^model_5/transformer_encoder_17/Encoder-SelfAttentionLayer-3/query/einsum/Einsum/ReadVariableOpU^model_5/transformer_encoder_17/Encoder-SelfAttentionLayer-3/value/add/ReadVariableOp_^model_5/transformer_encoder_17/Encoder-SelfAttentionLayer-3/value/einsum/Einsum/ReadVariableOp*
_output_shapes
 "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:���������P	:���������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2V
)model_5/FinalLayerNorm/add/ReadVariableOp)model_5/FinalLayerNorm/add/ReadVariableOp2V
)model_5/FinalLayerNorm/mul/ReadVariableOp)model_5/FinalLayerNorm/mul/ReadVariableOp2z
;model_5/FullyConnectedLayerAreaRatio/BiasAdd/ReadVariableOp;model_5/FullyConnectedLayerAreaRatio/BiasAdd/ReadVariableOp2x
:model_5/FullyConnectedLayerAreaRatio/MatMul/ReadVariableOp:model_5/FullyConnectedLayerAreaRatio/MatMul/ReadVariableOp2t
8model_5/FullyConnectedLayerSolved/BiasAdd/ReadVariableOp8model_5/FullyConnectedLayerSolved/BiasAdd/ReadVariableOp2r
7model_5/FullyConnectedLayerSolved/MatMul/ReadVariableOp7model_5/FullyConnectedLayerSolved/MatMul/ReadVariableOp2h
2model_5/PredictionAreaRatio/BiasAdd/ReadVariableOp2model_5/PredictionAreaRatio/BiasAdd/ReadVariableOp2f
1model_5/PredictionAreaRatio/MatMul/ReadVariableOp1model_5/PredictionAreaRatio/MatMul/ReadVariableOp2b
/model_5/PredictionSolved/BiasAdd/ReadVariableOp/model_5/PredictionSolved/BiasAdd/ReadVariableOp2`
.model_5/PredictionSolved/MatMul/ReadVariableOp.model_5/PredictionSolved/MatMul/ReadVariableOp2�
Rmodel_5/transformer_encoder_15/Encoder-1st-NormalizationLayer-1/add/ReadVariableOpRmodel_5/transformer_encoder_15/Encoder-1st-NormalizationLayer-1/add/ReadVariableOp2�
Rmodel_5/transformer_encoder_15/Encoder-1st-NormalizationLayer-1/mul/ReadVariableOpRmodel_5/transformer_encoder_15/Encoder-1st-NormalizationLayer-1/mul/ReadVariableOp2�
Rmodel_5/transformer_encoder_15/Encoder-2nd-NormalizationLayer-1/add/ReadVariableOpRmodel_5/transformer_encoder_15/Encoder-2nd-NormalizationLayer-1/add/ReadVariableOp2�
Rmodel_5/transformer_encoder_15/Encoder-2nd-NormalizationLayer-1/mul/ReadVariableOpRmodel_5/transformer_encoder_15/Encoder-2nd-NormalizationLayer-1/mul/ReadVariableOp2�
Rmodel_5/transformer_encoder_15/Encoder-FeedForwardLayer_1_1/BiasAdd/ReadVariableOpRmodel_5/transformer_encoder_15/Encoder-FeedForwardLayer_1_1/BiasAdd/ReadVariableOp2�
Tmodel_5/transformer_encoder_15/Encoder-FeedForwardLayer_1_1/Tensordot/ReadVariableOpTmodel_5/transformer_encoder_15/Encoder-FeedForwardLayer_1_1/Tensordot/ReadVariableOp2�
Rmodel_5/transformer_encoder_15/Encoder-FeedForwardLayer_2_1/BiasAdd/ReadVariableOpRmodel_5/transformer_encoder_15/Encoder-FeedForwardLayer_2_1/BiasAdd/ReadVariableOp2�
Tmodel_5/transformer_encoder_15/Encoder-FeedForwardLayer_2_1/Tensordot/ReadVariableOpTmodel_5/transformer_encoder_15/Encoder-FeedForwardLayer_2_1/Tensordot/ReadVariableOp2�
_model_5/transformer_encoder_15/Encoder-SelfAttentionLayer-1/attention_output/add/ReadVariableOp_model_5/transformer_encoder_15/Encoder-SelfAttentionLayer-1/attention_output/add/ReadVariableOp2�
imodel_5/transformer_encoder_15/Encoder-SelfAttentionLayer-1/attention_output/einsum/Einsum/ReadVariableOpimodel_5/transformer_encoder_15/Encoder-SelfAttentionLayer-1/attention_output/einsum/Einsum/ReadVariableOp2�
Rmodel_5/transformer_encoder_15/Encoder-SelfAttentionLayer-1/key/add/ReadVariableOpRmodel_5/transformer_encoder_15/Encoder-SelfAttentionLayer-1/key/add/ReadVariableOp2�
\model_5/transformer_encoder_15/Encoder-SelfAttentionLayer-1/key/einsum/Einsum/ReadVariableOp\model_5/transformer_encoder_15/Encoder-SelfAttentionLayer-1/key/einsum/Einsum/ReadVariableOp2�
Tmodel_5/transformer_encoder_15/Encoder-SelfAttentionLayer-1/query/add/ReadVariableOpTmodel_5/transformer_encoder_15/Encoder-SelfAttentionLayer-1/query/add/ReadVariableOp2�
^model_5/transformer_encoder_15/Encoder-SelfAttentionLayer-1/query/einsum/Einsum/ReadVariableOp^model_5/transformer_encoder_15/Encoder-SelfAttentionLayer-1/query/einsum/Einsum/ReadVariableOp2�
Tmodel_5/transformer_encoder_15/Encoder-SelfAttentionLayer-1/value/add/ReadVariableOpTmodel_5/transformer_encoder_15/Encoder-SelfAttentionLayer-1/value/add/ReadVariableOp2�
^model_5/transformer_encoder_15/Encoder-SelfAttentionLayer-1/value/einsum/Einsum/ReadVariableOp^model_5/transformer_encoder_15/Encoder-SelfAttentionLayer-1/value/einsum/Einsum/ReadVariableOp2�
Rmodel_5/transformer_encoder_16/Encoder-1st-NormalizationLayer-2/add/ReadVariableOpRmodel_5/transformer_encoder_16/Encoder-1st-NormalizationLayer-2/add/ReadVariableOp2�
Rmodel_5/transformer_encoder_16/Encoder-1st-NormalizationLayer-2/mul/ReadVariableOpRmodel_5/transformer_encoder_16/Encoder-1st-NormalizationLayer-2/mul/ReadVariableOp2�
Rmodel_5/transformer_encoder_16/Encoder-2nd-NormalizationLayer-2/add/ReadVariableOpRmodel_5/transformer_encoder_16/Encoder-2nd-NormalizationLayer-2/add/ReadVariableOp2�
Rmodel_5/transformer_encoder_16/Encoder-2nd-NormalizationLayer-2/mul/ReadVariableOpRmodel_5/transformer_encoder_16/Encoder-2nd-NormalizationLayer-2/mul/ReadVariableOp2�
Rmodel_5/transformer_encoder_16/Encoder-FeedForwardLayer_1_2/BiasAdd/ReadVariableOpRmodel_5/transformer_encoder_16/Encoder-FeedForwardLayer_1_2/BiasAdd/ReadVariableOp2�
Tmodel_5/transformer_encoder_16/Encoder-FeedForwardLayer_1_2/Tensordot/ReadVariableOpTmodel_5/transformer_encoder_16/Encoder-FeedForwardLayer_1_2/Tensordot/ReadVariableOp2�
Rmodel_5/transformer_encoder_16/Encoder-FeedForwardLayer_2_2/BiasAdd/ReadVariableOpRmodel_5/transformer_encoder_16/Encoder-FeedForwardLayer_2_2/BiasAdd/ReadVariableOp2�
Tmodel_5/transformer_encoder_16/Encoder-FeedForwardLayer_2_2/Tensordot/ReadVariableOpTmodel_5/transformer_encoder_16/Encoder-FeedForwardLayer_2_2/Tensordot/ReadVariableOp2�
_model_5/transformer_encoder_16/Encoder-SelfAttentionLayer-2/attention_output/add/ReadVariableOp_model_5/transformer_encoder_16/Encoder-SelfAttentionLayer-2/attention_output/add/ReadVariableOp2�
imodel_5/transformer_encoder_16/Encoder-SelfAttentionLayer-2/attention_output/einsum/Einsum/ReadVariableOpimodel_5/transformer_encoder_16/Encoder-SelfAttentionLayer-2/attention_output/einsum/Einsum/ReadVariableOp2�
Rmodel_5/transformer_encoder_16/Encoder-SelfAttentionLayer-2/key/add/ReadVariableOpRmodel_5/transformer_encoder_16/Encoder-SelfAttentionLayer-2/key/add/ReadVariableOp2�
\model_5/transformer_encoder_16/Encoder-SelfAttentionLayer-2/key/einsum/Einsum/ReadVariableOp\model_5/transformer_encoder_16/Encoder-SelfAttentionLayer-2/key/einsum/Einsum/ReadVariableOp2�
Tmodel_5/transformer_encoder_16/Encoder-SelfAttentionLayer-2/query/add/ReadVariableOpTmodel_5/transformer_encoder_16/Encoder-SelfAttentionLayer-2/query/add/ReadVariableOp2�
^model_5/transformer_encoder_16/Encoder-SelfAttentionLayer-2/query/einsum/Einsum/ReadVariableOp^model_5/transformer_encoder_16/Encoder-SelfAttentionLayer-2/query/einsum/Einsum/ReadVariableOp2�
Tmodel_5/transformer_encoder_16/Encoder-SelfAttentionLayer-2/value/add/ReadVariableOpTmodel_5/transformer_encoder_16/Encoder-SelfAttentionLayer-2/value/add/ReadVariableOp2�
^model_5/transformer_encoder_16/Encoder-SelfAttentionLayer-2/value/einsum/Einsum/ReadVariableOp^model_5/transformer_encoder_16/Encoder-SelfAttentionLayer-2/value/einsum/Einsum/ReadVariableOp2�
Rmodel_5/transformer_encoder_17/Encoder-1st-NormalizationLayer-3/add/ReadVariableOpRmodel_5/transformer_encoder_17/Encoder-1st-NormalizationLayer-3/add/ReadVariableOp2�
Rmodel_5/transformer_encoder_17/Encoder-1st-NormalizationLayer-3/mul/ReadVariableOpRmodel_5/transformer_encoder_17/Encoder-1st-NormalizationLayer-3/mul/ReadVariableOp2�
Rmodel_5/transformer_encoder_17/Encoder-2nd-NormalizationLayer-3/add/ReadVariableOpRmodel_5/transformer_encoder_17/Encoder-2nd-NormalizationLayer-3/add/ReadVariableOp2�
Rmodel_5/transformer_encoder_17/Encoder-2nd-NormalizationLayer-3/mul/ReadVariableOpRmodel_5/transformer_encoder_17/Encoder-2nd-NormalizationLayer-3/mul/ReadVariableOp2�
Rmodel_5/transformer_encoder_17/Encoder-FeedForwardLayer_1_3/BiasAdd/ReadVariableOpRmodel_5/transformer_encoder_17/Encoder-FeedForwardLayer_1_3/BiasAdd/ReadVariableOp2�
Tmodel_5/transformer_encoder_17/Encoder-FeedForwardLayer_1_3/Tensordot/ReadVariableOpTmodel_5/transformer_encoder_17/Encoder-FeedForwardLayer_1_3/Tensordot/ReadVariableOp2�
Rmodel_5/transformer_encoder_17/Encoder-FeedForwardLayer_2_3/BiasAdd/ReadVariableOpRmodel_5/transformer_encoder_17/Encoder-FeedForwardLayer_2_3/BiasAdd/ReadVariableOp2�
Tmodel_5/transformer_encoder_17/Encoder-FeedForwardLayer_2_3/Tensordot/ReadVariableOpTmodel_5/transformer_encoder_17/Encoder-FeedForwardLayer_2_3/Tensordot/ReadVariableOp2�
_model_5/transformer_encoder_17/Encoder-SelfAttentionLayer-3/attention_output/add/ReadVariableOp_model_5/transformer_encoder_17/Encoder-SelfAttentionLayer-3/attention_output/add/ReadVariableOp2�
imodel_5/transformer_encoder_17/Encoder-SelfAttentionLayer-3/attention_output/einsum/Einsum/ReadVariableOpimodel_5/transformer_encoder_17/Encoder-SelfAttentionLayer-3/attention_output/einsum/Einsum/ReadVariableOp2�
Rmodel_5/transformer_encoder_17/Encoder-SelfAttentionLayer-3/key/add/ReadVariableOpRmodel_5/transformer_encoder_17/Encoder-SelfAttentionLayer-3/key/add/ReadVariableOp2�
\model_5/transformer_encoder_17/Encoder-SelfAttentionLayer-3/key/einsum/Einsum/ReadVariableOp\model_5/transformer_encoder_17/Encoder-SelfAttentionLayer-3/key/einsum/Einsum/ReadVariableOp2�
Tmodel_5/transformer_encoder_17/Encoder-SelfAttentionLayer-3/query/add/ReadVariableOpTmodel_5/transformer_encoder_17/Encoder-SelfAttentionLayer-3/query/add/ReadVariableOp2�
^model_5/transformer_encoder_17/Encoder-SelfAttentionLayer-3/query/einsum/Einsum/ReadVariableOp^model_5/transformer_encoder_17/Encoder-SelfAttentionLayer-3/query/einsum/Einsum/ReadVariableOp2�
Tmodel_5/transformer_encoder_17/Encoder-SelfAttentionLayer-3/value/add/ReadVariableOpTmodel_5/transformer_encoder_17/Encoder-SelfAttentionLayer-3/value/add/ReadVariableOp2�
^model_5/transformer_encoder_17/Encoder-SelfAttentionLayer-3/value/einsum/Einsum/ReadVariableOp^model_5/transformer_encoder_17/Encoder-SelfAttentionLayer-3/value/einsum/Einsum/ReadVariableOp:(;$
"
_user_specified_name
resource:(:$
"
_user_specified_name
resource:(9$
"
_user_specified_name
resource:(8$
"
_user_specified_name
resource:(7$
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
�
�
Y__inference_FullyConnectedLayerAreaRatio_layer_call_and_return_conditional_losses_1432868

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
�
�
Y__inference_FullyConnectedLayerAreaRatio_layer_call_and_return_conditional_losses_1429964

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
�/
�
)__inference_model_5_layer_call_fn_1430944
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



unknown_52:


unknown_53:


unknown_54:

unknown_55:


unknown_56:
identity

identity_1��StatefulPartitionedCall�
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
unknown_52
unknown_53
unknown_54
unknown_55
unknown_56*G
Tin@
>2<*
Tout
2*
_collective_manager_ids
 *
_output_shapes

::*\
_read_only_resource_inputs>
<:	
 !"#$%&'()*+,-./0123456789:;*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_model_5_layer_call_and_return_conditional_losses_1430696`
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes
:b

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*
_output_shapes
:<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:���������P	:���������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:';#
!
_user_specified_name	1430938:':#
!
_user_specified_name	1430936:'9#
!
_user_specified_name	1430934:'8#
!
_user_specified_name	1430932:'7#
!
_user_specified_name	1430930:'6#
!
_user_specified_name	1430928:'5#
!
_user_specified_name	1430926:'4#
!
_user_specified_name	1430924:'3#
!
_user_specified_name	1430922:'2#
!
_user_specified_name	1430920:'1#
!
_user_specified_name	1430918:'0#
!
_user_specified_name	1430916:'/#
!
_user_specified_name	1430914:'.#
!
_user_specified_name	1430912:'-#
!
_user_specified_name	1430910:',#
!
_user_specified_name	1430908:'+#
!
_user_specified_name	1430906:'*#
!
_user_specified_name	1430904:')#
!
_user_specified_name	1430902:'(#
!
_user_specified_name	1430900:''#
!
_user_specified_name	1430898:'&#
!
_user_specified_name	1430896:'%#
!
_user_specified_name	1430894:'$#
!
_user_specified_name	1430892:'##
!
_user_specified_name	1430890:'"#
!
_user_specified_name	1430888:'!#
!
_user_specified_name	1430886:' #
!
_user_specified_name	1430884:'#
!
_user_specified_name	1430882:'#
!
_user_specified_name	1430880:'#
!
_user_specified_name	1430878:'#
!
_user_specified_name	1430876:'#
!
_user_specified_name	1430874:'#
!
_user_specified_name	1430872:'#
!
_user_specified_name	1430870:'#
!
_user_specified_name	1430868:'#
!
_user_specified_name	1430866:'#
!
_user_specified_name	1430864:'#
!
_user_specified_name	1430862:'#
!
_user_specified_name	1430860:'#
!
_user_specified_name	1430858:'#
!
_user_specified_name	1430856:'#
!
_user_specified_name	1430854:'#
!
_user_specified_name	1430852:'#
!
_user_specified_name	1430850:'#
!
_user_specified_name	1430848:'#
!
_user_specified_name	1430846:'#
!
_user_specified_name	1430844:'#
!
_user_specified_name	1430842:'
#
!
_user_specified_name	1430840:'	#
!
_user_specified_name	1430838:'#
!
_user_specified_name	1430836:'#
!
_user_specified_name	1430834:'#
!
_user_specified_name	1430832:'#
!
_user_specified_name	1430830:'#
!
_user_specified_name	1430828:'#
!
_user_specified_name	1430826:'#
!
_user_specified_name	1430824:WS
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
�

�
M__inference_PredictionSolved_layer_call_and_return_conditional_losses_1429980

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
�
y
M__inference_ConcatenateLayer_layer_call_and_return_conditional_losses_1432841
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
�
_
C__inference_Output_layer_call_and_return_conditional_losses_1432950

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
S__inference_transformer_encoder_15_layer_call_and_return_conditional_losses_1431671

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
:���������P	]
dropout_15/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?�
dropout_15/dropout/MulMul-Encoder-FeedForwardLayer_2_1/BiasAdd:output:0!dropout_15/dropout/Const:output:0*
T0*+
_output_shapes
:���������P	�
dropout_15/dropout/ShapeShape-Encoder-FeedForwardLayer_2_1/BiasAdd:output:0*
T0*
_output_shapes
::���
/dropout_15/dropout/random_uniform/RandomUniformRandomUniform!dropout_15/dropout/Shape:output:0*
T0*+
_output_shapes
:���������P	*
dtype0*
seed2*
seed��f
!dropout_15/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=�
dropout_15/dropout/GreaterEqualGreaterEqual8dropout_15/dropout/random_uniform/RandomUniform:output:0*dropout_15/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:���������P	_
dropout_15/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout_15/dropout/SelectV2SelectV2#dropout_15/dropout/GreaterEqual:z:0dropout_15/dropout/Mul:z:0#dropout_15/dropout/Const_1:output:0*
T0*+
_output_shapes
:���������P	�
Encoder-2nd-AdditionLayer-1/addAddV2#Encoder-1st-AdditionLayer-1/add:z:0$dropout_15/dropout/SelectV2:output:0*
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
�
�
8__inference_transformer_encoder_17_layer_call_fn_1432324

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
GPU2*0J 8� *\
fWRU
S__inference_transformer_encoder_17_layer_call_and_return_conditional_losses_1429814s
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
StatefulPartitionedCallStatefulPartitionedCall:'#
!
_user_specified_name	1432318:'#
!
_user_specified_name	1432316:'#
!
_user_specified_name	1432314:'#
!
_user_specified_name	1432312:'#
!
_user_specified_name	1432310:'#
!
_user_specified_name	1432308:'
#
!
_user_specified_name	1432306:'	#
!
_user_specified_name	1432304:'#
!
_user_specified_name	1432302:'#
!
_user_specified_name	1432300:'#
!
_user_specified_name	1432298:'#
!
_user_specified_name	1432296:'#
!
_user_specified_name	1432294:'#
!
_user_specified_name	1432292:'#
!
_user_specified_name	1432290:'#
!
_user_specified_name	1432288:S O
+
_output_shapes
:���������P	
 
_user_specified_nameinputs
��
�
S__inference_transformer_encoder_15_layer_call_and_return_conditional_losses_1429370

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
:���������P	]
dropout_15/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?�
dropout_15/dropout/MulMul-Encoder-FeedForwardLayer_2_1/BiasAdd:output:0!dropout_15/dropout/Const:output:0*
T0*+
_output_shapes
:���������P	�
dropout_15/dropout/ShapeShape-Encoder-FeedForwardLayer_2_1/BiasAdd:output:0*
T0*
_output_shapes
::���
/dropout_15/dropout/random_uniform/RandomUniformRandomUniform!dropout_15/dropout/Shape:output:0*
T0*+
_output_shapes
:���������P	*
dtype0*
seed2*
seed��f
!dropout_15/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=�
dropout_15/dropout/GreaterEqualGreaterEqual8dropout_15/dropout/random_uniform/RandomUniform:output:0*dropout_15/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:���������P	_
dropout_15/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout_15/dropout/SelectV2SelectV2#dropout_15/dropout/GreaterEqual:z:0dropout_15/dropout/Mul:z:0#dropout_15/dropout/Const_1:output:0*
T0*+
_output_shapes
:���������P	�
Encoder-2nd-AdditionLayer-1/addAddV2#Encoder-1st-AdditionLayer-1/add:z:0$dropout_15/dropout/SelectV2:output:0*
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
m
Q__inference_StandardizeTimeLimit_layer_call_and_return_conditional_losses_1429913

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
�{
�
D__inference_model_5_layer_call_and_return_conditional_losses_1430696
stacklevelinputfeatures
timelimitinput,
transformer_encoder_15_1430191:	,
transformer_encoder_15_1430193:	4
transformer_encoder_15_1430195:	0
transformer_encoder_15_1430197:4
transformer_encoder_15_1430199:	0
transformer_encoder_15_1430201:4
transformer_encoder_15_1430203:	0
transformer_encoder_15_1430205:4
transformer_encoder_15_1430207:	,
transformer_encoder_15_1430209:	,
transformer_encoder_15_1430211:	,
transformer_encoder_15_1430213:	0
transformer_encoder_15_1430215:		,
transformer_encoder_15_1430217:	0
transformer_encoder_15_1430219:		,
transformer_encoder_15_1430221:	,
transformer_encoder_16_1430399:	,
transformer_encoder_16_1430401:	4
transformer_encoder_16_1430403:	0
transformer_encoder_16_1430405:4
transformer_encoder_16_1430407:	0
transformer_encoder_16_1430409:4
transformer_encoder_16_1430411:	0
transformer_encoder_16_1430413:4
transformer_encoder_16_1430415:	,
transformer_encoder_16_1430417:	,
transformer_encoder_16_1430419:	,
transformer_encoder_16_1430421:	0
transformer_encoder_16_1430423:		,
transformer_encoder_16_1430425:	0
transformer_encoder_16_1430427:		,
transformer_encoder_16_1430429:	,
transformer_encoder_17_1430607:	,
transformer_encoder_17_1430609:	4
transformer_encoder_17_1430611:	0
transformer_encoder_17_1430613:4
transformer_encoder_17_1430615:	0
transformer_encoder_17_1430617:4
transformer_encoder_17_1430619:	0
transformer_encoder_17_1430621:4
transformer_encoder_17_1430623:	,
transformer_encoder_17_1430625:	,
transformer_encoder_17_1430627:	,
transformer_encoder_17_1430629:	0
transformer_encoder_17_1430631:		,
transformer_encoder_17_1430633:	0
transformer_encoder_17_1430635:		,
transformer_encoder_17_1430637:	$
finallayernorm_1430641:	$
finallayernorm_1430643:	3
!fullyconnectedlayersolved_1430667:

/
!fullyconnectedlayersolved_1430669:
6
$fullyconnectedlayerarearatio_1430672:

2
$fullyconnectedlayerarearatio_1430674:
*
predictionsolved_1430677:
&
predictionsolved_1430679:-
predictionarearatio_1430682:
)
predictionarearatio_1430684:
identity

identity_1��&FinalLayerNorm/StatefulPartitionedCall�4FullyConnectedLayerAreaRatio/StatefulPartitionedCall�1FullyConnectedLayerSolved/StatefulPartitionedCall�+PredictionAreaRatio/StatefulPartitionedCall�(PredictionSolved/StatefulPartitionedCall�.transformer_encoder_15/StatefulPartitionedCall�.transformer_encoder_16/StatefulPartitionedCall�.transformer_encoder_17/StatefulPartitionedCall�
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
GPU2*0J 8� *R
fMRK
I__inference_MaskingLayer_layer_call_and_return_conditional_losses_1429180�
transformer_encoder_15/CastCast%MaskingLayer/PartitionedCall:output:0*

DstT0*

SrcT0*+
_output_shapes
:���������P	�
.transformer_encoder_15/StatefulPartitionedCallStatefulPartitionedCalltransformer_encoder_15/Cast:y:0transformer_encoder_15_1430191transformer_encoder_15_1430193transformer_encoder_15_1430195transformer_encoder_15_1430197transformer_encoder_15_1430199transformer_encoder_15_1430201transformer_encoder_15_1430203transformer_encoder_15_1430205transformer_encoder_15_1430207transformer_encoder_15_1430209transformer_encoder_15_1430211transformer_encoder_15_1430213transformer_encoder_15_1430215transformer_encoder_15_1430217transformer_encoder_15_1430219transformer_encoder_15_1430221*
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
GPU2*0J 8� *\
fWRU
S__inference_transformer_encoder_15_layer_call_and_return_conditional_losses_1430190�
.transformer_encoder_16/StatefulPartitionedCallStatefulPartitionedCall7transformer_encoder_15/StatefulPartitionedCall:output:0transformer_encoder_16_1430399transformer_encoder_16_1430401transformer_encoder_16_1430403transformer_encoder_16_1430405transformer_encoder_16_1430407transformer_encoder_16_1430409transformer_encoder_16_1430411transformer_encoder_16_1430413transformer_encoder_16_1430415transformer_encoder_16_1430417transformer_encoder_16_1430419transformer_encoder_16_1430421transformer_encoder_16_1430423transformer_encoder_16_1430425transformer_encoder_16_1430427transformer_encoder_16_1430429*
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
GPU2*0J 8� *\
fWRU
S__inference_transformer_encoder_16_layer_call_and_return_conditional_losses_1430398�
.transformer_encoder_17/StatefulPartitionedCallStatefulPartitionedCall7transformer_encoder_16/StatefulPartitionedCall:output:0transformer_encoder_17_1430607transformer_encoder_17_1430609transformer_encoder_17_1430611transformer_encoder_17_1430613transformer_encoder_17_1430615transformer_encoder_17_1430617transformer_encoder_17_1430619transformer_encoder_17_1430621transformer_encoder_17_1430623transformer_encoder_17_1430625transformer_encoder_17_1430627transformer_encoder_17_1430629transformer_encoder_17_1430631transformer_encoder_17_1430633transformer_encoder_17_1430635transformer_encoder_17_1430637*
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
GPU2*0J 8� *\
fWRU
S__inference_transformer_encoder_17_layer_call_and_return_conditional_losses_1430606�
&FinalLayerNorm/StatefulPartitionedCallStatefulPartitionedCall7transformer_encoder_17/StatefulPartitionedCall:output:0finallayernorm_1430641finallayernorm_1430643*
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
GPU2*0J 8� *T
fORM
K__inference_FinalLayerNorm_layer_call_and_return_conditional_losses_1429890�
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
GPU2*0J 8� *f
faR_
]__inference_ReduceStackDimensionViaSummation_layer_call_and_return_conditional_losses_1430653r
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
GPU2*0J 8� *Z
fURS
Q__inference_StandardizeTimeLimit_layer_call_and_return_conditional_losses_1430663�
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
GPU2*0J 8� *V
fQRO
M__inference_ConcatenateLayer_layer_call_and_return_conditional_losses_1429921�
"ConcatenateLayer/PartitionedCall_1PartitionedCall9ReduceStackDimensionViaSummation/PartitionedCall:output:0-StandardizeTimeLimit/PartitionedCall:output:0*
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
GPU2*0J 8� *V
fQRO
M__inference_ConcatenateLayer_layer_call_and_return_conditional_losses_1429921�
1FullyConnectedLayerSolved/StatefulPartitionedCallStatefulPartitionedCall)ConcatenateLayer/PartitionedCall:output:0!fullyconnectedlayersolved_1430667!fullyconnectedlayersolved_1430669*
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
GPU2*0J 8� *_
fZRX
V__inference_FullyConnectedLayerSolved_layer_call_and_return_conditional_losses_1429941�
4FullyConnectedLayerAreaRatio/StatefulPartitionedCallStatefulPartitionedCall+ConcatenateLayer/PartitionedCall_1:output:0$fullyconnectedlayerarearatio_1430672$fullyconnectedlayerarearatio_1430674*
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
GPU2*0J 8� *b
f]R[
Y__inference_FullyConnectedLayerAreaRatio_layer_call_and_return_conditional_losses_1429964�
(PredictionSolved/StatefulPartitionedCallStatefulPartitionedCall:FullyConnectedLayerSolved/StatefulPartitionedCall:output:0predictionsolved_1430677predictionsolved_1430679*
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
GPU2*0J 8� *V
fQRO
M__inference_PredictionSolved_layer_call_and_return_conditional_losses_1429980�
+PredictionAreaRatio/StatefulPartitionedCallStatefulPartitionedCall=FullyConnectedLayerAreaRatio/StatefulPartitionedCall:output:0predictionarearatio_1430682predictionarearatio_1430684*
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
GPU2*0J 8� *Y
fTRR
P__inference_PredictionAreaRatio_layer_call_and_return_conditional_losses_1429996�
Output/PartitionedCallPartitionedCall1PredictionSolved/StatefulPartitionedCall:output:0*
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
GPU2*0J 8� *L
fGRE
C__inference_Output_layer_call_and_return_conditional_losses_1430691�
Output/PartitionedCall_1PartitionedCall4PredictionAreaRatio/StatefulPartitionedCall:output:0*
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
GPU2*0J 8� *L
fGRE
C__inference_Output_layer_call_and_return_conditional_losses_1430691a
IdentityIdentity!Output/PartitionedCall_1:output:0^NoOp*
T0*
_output_shapes
:a

Identity_1IdentityOutput/PartitionedCall:output:0^NoOp*
T0*
_output_shapes
:�
NoOpNoOp'^FinalLayerNorm/StatefulPartitionedCall5^FullyConnectedLayerAreaRatio/StatefulPartitionedCall2^FullyConnectedLayerSolved/StatefulPartitionedCall,^PredictionAreaRatio/StatefulPartitionedCall)^PredictionSolved/StatefulPartitionedCall/^transformer_encoder_15/StatefulPartitionedCall/^transformer_encoder_16/StatefulPartitionedCall/^transformer_encoder_17/StatefulPartitionedCall*
_output_shapes
 "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:���������P	:���������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2P
&FinalLayerNorm/StatefulPartitionedCall&FinalLayerNorm/StatefulPartitionedCall2l
4FullyConnectedLayerAreaRatio/StatefulPartitionedCall4FullyConnectedLayerAreaRatio/StatefulPartitionedCall2f
1FullyConnectedLayerSolved/StatefulPartitionedCall1FullyConnectedLayerSolved/StatefulPartitionedCall2Z
+PredictionAreaRatio/StatefulPartitionedCall+PredictionAreaRatio/StatefulPartitionedCall2T
(PredictionSolved/StatefulPartitionedCall(PredictionSolved/StatefulPartitionedCall2`
.transformer_encoder_15/StatefulPartitionedCall.transformer_encoder_15/StatefulPartitionedCall2`
.transformer_encoder_16/StatefulPartitionedCall.transformer_encoder_16/StatefulPartitionedCall2`
.transformer_encoder_17/StatefulPartitionedCall.transformer_encoder_17/StatefulPartitionedCall:';#
!
_user_specified_name	1430684:':#
!
_user_specified_name	1430682:'9#
!
_user_specified_name	1430679:'8#
!
_user_specified_name	1430677:'7#
!
_user_specified_name	1430674:'6#
!
_user_specified_name	1430672:'5#
!
_user_specified_name	1430669:'4#
!
_user_specified_name	1430667:'3#
!
_user_specified_name	1430643:'2#
!
_user_specified_name	1430641:'1#
!
_user_specified_name	1430637:'0#
!
_user_specified_name	1430635:'/#
!
_user_specified_name	1430633:'.#
!
_user_specified_name	1430631:'-#
!
_user_specified_name	1430629:',#
!
_user_specified_name	1430627:'+#
!
_user_specified_name	1430625:'*#
!
_user_specified_name	1430623:')#
!
_user_specified_name	1430621:'(#
!
_user_specified_name	1430619:''#
!
_user_specified_name	1430617:'&#
!
_user_specified_name	1430615:'%#
!
_user_specified_name	1430613:'$#
!
_user_specified_name	1430611:'##
!
_user_specified_name	1430609:'"#
!
_user_specified_name	1430607:'!#
!
_user_specified_name	1430429:' #
!
_user_specified_name	1430427:'#
!
_user_specified_name	1430425:'#
!
_user_specified_name	1430423:'#
!
_user_specified_name	1430421:'#
!
_user_specified_name	1430419:'#
!
_user_specified_name	1430417:'#
!
_user_specified_name	1430415:'#
!
_user_specified_name	1430413:'#
!
_user_specified_name	1430411:'#
!
_user_specified_name	1430409:'#
!
_user_specified_name	1430407:'#
!
_user_specified_name	1430405:'#
!
_user_specified_name	1430403:'#
!
_user_specified_name	1430401:'#
!
_user_specified_name	1430399:'#
!
_user_specified_name	1430221:'#
!
_user_specified_name	1430219:'#
!
_user_specified_name	1430217:'#
!
_user_specified_name	1430215:'#
!
_user_specified_name	1430213:'#
!
_user_specified_name	1430211:'#
!
_user_specified_name	1430209:'
#
!
_user_specified_name	1430207:'	#
!
_user_specified_name	1430205:'#
!
_user_specified_name	1430203:'#
!
_user_specified_name	1430201:'#
!
_user_specified_name	1430199:'#
!
_user_specified_name	1430197:'#
!
_user_specified_name	1430195:'#
!
_user_specified_name	1430193:'#
!
_user_specified_name	1430191:WS
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
� 
�
K__inference_FinalLayerNorm_layer_call_and_return_conditional_losses_1432776

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
�
_
C__inference_Output_layer_call_and_return_conditional_losses_1430006

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
�
�
;__inference_FullyConnectedLayerSolved_layer_call_fn_1432877

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
GPU2*0J 8� *_
fZRX
V__inference_FullyConnectedLayerSolved_layer_call_and_return_conditional_losses_1429941o
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
StatefulPartitionedCallStatefulPartitionedCall:'#
!
_user_specified_name	1432873:'#
!
_user_specified_name	1432871:O K
'
_output_shapes
:���������

 
_user_specified_nameinputs
��
�
S__inference_transformer_encoder_16_layer_call_and_return_conditional_losses_1432285

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
dropout_16/IdentityIdentity-Encoder-FeedForwardLayer_2_2/BiasAdd:output:0*
T0*+
_output_shapes
:���������P	�
Encoder-2nd-AdditionLayer-2/addAddV2#Encoder-1st-AdditionLayer-2/add:z:0dropout_16/Identity:output:0*
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
�
y
]__inference_ReduceStackDimensionViaSummation_layer_call_and_return_conditional_losses_1432794

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
�
�
8__inference_transformer_encoder_15_layer_call_fn_1431444

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
GPU2*0J 8� *\
fWRU
S__inference_transformer_encoder_15_layer_call_and_return_conditional_losses_1429370s
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
StatefulPartitionedCallStatefulPartitionedCall:'#
!
_user_specified_name	1431438:'#
!
_user_specified_name	1431436:'#
!
_user_specified_name	1431434:'#
!
_user_specified_name	1431432:'#
!
_user_specified_name	1431430:'#
!
_user_specified_name	1431428:'
#
!
_user_specified_name	1431426:'	#
!
_user_specified_name	1431424:'#
!
_user_specified_name	1431422:'#
!
_user_specified_name	1431420:'#
!
_user_specified_name	1431418:'#
!
_user_specified_name	1431416:'#
!
_user_specified_name	1431414:'#
!
_user_specified_name	1431412:'#
!
_user_specified_name	1431410:'#
!
_user_specified_name	1431408:S O
+
_output_shapes
:���������P	
 
_user_specified_nameinputs
�/
�
%__inference_signature_wrapper_1431389
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



unknown_52:


unknown_53:


unknown_54:

unknown_55:


unknown_56:
identity

identity_1��StatefulPartitionedCall�
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
unknown_52
unknown_53
unknown_54
unknown_55
unknown_56*G
Tin@
>2<*
Tout
2*
_collective_manager_ids
 *
_output_shapes

::*\
_read_only_resource_inputs>
<:	
 !"#$%&'()*+,-./0123456789:;*0
config_proto 

CPU

GPU2*0J 8� *+
f&R$
"__inference__wrapped_model_1429166`
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes
:b

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*
_output_shapes
:<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:���������P	:���������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:';#
!
_user_specified_name	1431383:':#
!
_user_specified_name	1431381:'9#
!
_user_specified_name	1431379:'8#
!
_user_specified_name	1431377:'7#
!
_user_specified_name	1431375:'6#
!
_user_specified_name	1431373:'5#
!
_user_specified_name	1431371:'4#
!
_user_specified_name	1431369:'3#
!
_user_specified_name	1431367:'2#
!
_user_specified_name	1431365:'1#
!
_user_specified_name	1431363:'0#
!
_user_specified_name	1431361:'/#
!
_user_specified_name	1431359:'.#
!
_user_specified_name	1431357:'-#
!
_user_specified_name	1431355:',#
!
_user_specified_name	1431353:'+#
!
_user_specified_name	1431351:'*#
!
_user_specified_name	1431349:')#
!
_user_specified_name	1431347:'(#
!
_user_specified_name	1431345:''#
!
_user_specified_name	1431343:'&#
!
_user_specified_name	1431341:'%#
!
_user_specified_name	1431339:'$#
!
_user_specified_name	1431337:'##
!
_user_specified_name	1431335:'"#
!
_user_specified_name	1431333:'!#
!
_user_specified_name	1431331:' #
!
_user_specified_name	1431329:'#
!
_user_specified_name	1431327:'#
!
_user_specified_name	1431325:'#
!
_user_specified_name	1431323:'#
!
_user_specified_name	1431321:'#
!
_user_specified_name	1431319:'#
!
_user_specified_name	1431317:'#
!
_user_specified_name	1431315:'#
!
_user_specified_name	1431313:'#
!
_user_specified_name	1431311:'#
!
_user_specified_name	1431309:'#
!
_user_specified_name	1431307:'#
!
_user_specified_name	1431305:'#
!
_user_specified_name	1431303:'#
!
_user_specified_name	1431301:'#
!
_user_specified_name	1431299:'#
!
_user_specified_name	1431297:'#
!
_user_specified_name	1431295:'#
!
_user_specified_name	1431293:'#
!
_user_specified_name	1431291:'#
!
_user_specified_name	1431289:'#
!
_user_specified_name	1431287:'
#
!
_user_specified_name	1431285:'	#
!
_user_specified_name	1431283:'#
!
_user_specified_name	1431281:'#
!
_user_specified_name	1431279:'#
!
_user_specified_name	1431277:'#
!
_user_specified_name	1431275:'#
!
_user_specified_name	1431273:'#
!
_user_specified_name	1431271:'#
!
_user_specified_name	1431269:WS
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
m
Q__inference_StandardizeTimeLimit_layer_call_and_return_conditional_losses_1432820

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
�
2__inference_PredictionSolved_layer_call_fn_1432924

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
GPU2*0J 8� *V
fQRO
M__inference_PredictionSolved_layer_call_and_return_conditional_losses_1429980o
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
StatefulPartitionedCallStatefulPartitionedCall:'#
!
_user_specified_name	1432920:'#
!
_user_specified_name	1432918:O K
'
_output_shapes
:���������

 
_user_specified_nameinputs
��
�F
 __inference__traced_save_1433327
file_prefix9
+read_disablecopyonread_finallayernorm_gamma:	:
,read_1_disablecopyonread_finallayernorm_beta:	N
<read_2_disablecopyonread_fullyconnectedlayerarearatio_kernel:

H
:read_3_disablecopyonread_fullyconnectedlayerarearatio_bias:
K
9read_4_disablecopyonread_fullyconnectedlayersolved_kernel:

E
7read_5_disablecopyonread_fullyconnectedlayersolved_bias:
E
3read_6_disablecopyonread_predictionarearatio_kernel:
?
1read_7_disablecopyonread_predictionarearatio_bias:B
0read_8_disablecopyonread_predictionsolved_kernel:
<
.read_9_disablecopyonread_predictionsolved_bias:p
Zread_10_disablecopyonread_transformer_encoder_15_encoder_selfattentionlayer_1_query_kernel:	j
Xread_11_disablecopyonread_transformer_encoder_15_encoder_selfattentionlayer_1_query_bias:n
Xread_12_disablecopyonread_transformer_encoder_15_encoder_selfattentionlayer_1_key_kernel:	h
Vread_13_disablecopyonread_transformer_encoder_15_encoder_selfattentionlayer_1_key_bias:p
Zread_14_disablecopyonread_transformer_encoder_15_encoder_selfattentionlayer_1_value_kernel:	j
Xread_15_disablecopyonread_transformer_encoder_15_encoder_selfattentionlayer_1_value_bias:{
eread_16_disablecopyonread_transformer_encoder_15_encoder_selfattentionlayer_1_attention_output_kernel:	q
cread_17_disablecopyonread_transformer_encoder_15_encoder_selfattentionlayer_1_attention_output_bias:	e
Wread_18_disablecopyonread_transformer_encoder_15_encoder_1st_normalizationlayer_1_gamma:	d
Vread_19_disablecopyonread_transformer_encoder_15_encoder_1st_normalizationlayer_1_beta:	e
Wread_20_disablecopyonread_transformer_encoder_15_encoder_2nd_normalizationlayer_1_gamma:	d
Vread_21_disablecopyonread_transformer_encoder_15_encoder_2nd_normalizationlayer_1_beta:	f
Tread_22_disablecopyonread_transformer_encoder_15_encoder_feedforwardlayer_1_1_kernel:		`
Rread_23_disablecopyonread_transformer_encoder_15_encoder_feedforwardlayer_1_1_bias:	f
Tread_24_disablecopyonread_transformer_encoder_15_encoder_feedforwardlayer_2_1_kernel:		`
Rread_25_disablecopyonread_transformer_encoder_15_encoder_feedforwardlayer_2_1_bias:	p
Zread_26_disablecopyonread_transformer_encoder_16_encoder_selfattentionlayer_2_query_kernel:	j
Xread_27_disablecopyonread_transformer_encoder_16_encoder_selfattentionlayer_2_query_bias:n
Xread_28_disablecopyonread_transformer_encoder_16_encoder_selfattentionlayer_2_key_kernel:	h
Vread_29_disablecopyonread_transformer_encoder_16_encoder_selfattentionlayer_2_key_bias:p
Zread_30_disablecopyonread_transformer_encoder_16_encoder_selfattentionlayer_2_value_kernel:	j
Xread_31_disablecopyonread_transformer_encoder_16_encoder_selfattentionlayer_2_value_bias:{
eread_32_disablecopyonread_transformer_encoder_16_encoder_selfattentionlayer_2_attention_output_kernel:	q
cread_33_disablecopyonread_transformer_encoder_16_encoder_selfattentionlayer_2_attention_output_bias:	e
Wread_34_disablecopyonread_transformer_encoder_16_encoder_1st_normalizationlayer_2_gamma:	d
Vread_35_disablecopyonread_transformer_encoder_16_encoder_1st_normalizationlayer_2_beta:	e
Wread_36_disablecopyonread_transformer_encoder_16_encoder_2nd_normalizationlayer_2_gamma:	d
Vread_37_disablecopyonread_transformer_encoder_16_encoder_2nd_normalizationlayer_2_beta:	f
Tread_38_disablecopyonread_transformer_encoder_16_encoder_feedforwardlayer_1_2_kernel:		`
Rread_39_disablecopyonread_transformer_encoder_16_encoder_feedforwardlayer_1_2_bias:	f
Tread_40_disablecopyonread_transformer_encoder_16_encoder_feedforwardlayer_2_2_kernel:		`
Rread_41_disablecopyonread_transformer_encoder_16_encoder_feedforwardlayer_2_2_bias:	p
Zread_42_disablecopyonread_transformer_encoder_17_encoder_selfattentionlayer_3_query_kernel:	j
Xread_43_disablecopyonread_transformer_encoder_17_encoder_selfattentionlayer_3_query_bias:n
Xread_44_disablecopyonread_transformer_encoder_17_encoder_selfattentionlayer_3_key_kernel:	h
Vread_45_disablecopyonread_transformer_encoder_17_encoder_selfattentionlayer_3_key_bias:p
Zread_46_disablecopyonread_transformer_encoder_17_encoder_selfattentionlayer_3_value_kernel:	j
Xread_47_disablecopyonread_transformer_encoder_17_encoder_selfattentionlayer_3_value_bias:{
eread_48_disablecopyonread_transformer_encoder_17_encoder_selfattentionlayer_3_attention_output_kernel:	q
cread_49_disablecopyonread_transformer_encoder_17_encoder_selfattentionlayer_3_attention_output_bias:	e
Wread_50_disablecopyonread_transformer_encoder_17_encoder_1st_normalizationlayer_3_gamma:	d
Vread_51_disablecopyonread_transformer_encoder_17_encoder_1st_normalizationlayer_3_beta:	e
Wread_52_disablecopyonread_transformer_encoder_17_encoder_2nd_normalizationlayer_3_gamma:	d
Vread_53_disablecopyonread_transformer_encoder_17_encoder_2nd_normalizationlayer_3_beta:	f
Tread_54_disablecopyonread_transformer_encoder_17_encoder_feedforwardlayer_1_3_kernel:		`
Rread_55_disablecopyonread_transformer_encoder_17_encoder_feedforwardlayer_1_3_bias:	f
Tread_56_disablecopyonread_transformer_encoder_17_encoder_feedforwardlayer_2_3_kernel:		`
Rread_57_disablecopyonread_transformer_encoder_17_encoder_feedforwardlayer_2_3_bias:	
savev2_const
identity_117��MergeV2Checkpoints�Read/DisableCopyOnRead�Read/ReadVariableOp�Read_1/DisableCopyOnRead�Read_1/ReadVariableOp�Read_10/DisableCopyOnRead�Read_10/ReadVariableOp�Read_11/DisableCopyOnRead�Read_11/ReadVariableOp�Read_12/DisableCopyOnRead�Read_12/ReadVariableOp�Read_13/DisableCopyOnRead�Read_13/ReadVariableOp�Read_14/DisableCopyOnRead�Read_14/ReadVariableOp�Read_15/DisableCopyOnRead�Read_15/ReadVariableOp�Read_16/DisableCopyOnRead�Read_16/ReadVariableOp�Read_17/DisableCopyOnRead�Read_17/ReadVariableOp�Read_18/DisableCopyOnRead�Read_18/ReadVariableOp�Read_19/DisableCopyOnRead�Read_19/ReadVariableOp�Read_2/DisableCopyOnRead�Read_2/ReadVariableOp�Read_20/DisableCopyOnRead�Read_20/ReadVariableOp�Read_21/DisableCopyOnRead�Read_21/ReadVariableOp�Read_22/DisableCopyOnRead�Read_22/ReadVariableOp�Read_23/DisableCopyOnRead�Read_23/ReadVariableOp�Read_24/DisableCopyOnRead�Read_24/ReadVariableOp�Read_25/DisableCopyOnRead�Read_25/ReadVariableOp�Read_26/DisableCopyOnRead�Read_26/ReadVariableOp�Read_27/DisableCopyOnRead�Read_27/ReadVariableOp�Read_28/DisableCopyOnRead�Read_28/ReadVariableOp�Read_29/DisableCopyOnRead�Read_29/ReadVariableOp�Read_3/DisableCopyOnRead�Read_3/ReadVariableOp�Read_30/DisableCopyOnRead�Read_30/ReadVariableOp�Read_31/DisableCopyOnRead�Read_31/ReadVariableOp�Read_32/DisableCopyOnRead�Read_32/ReadVariableOp�Read_33/DisableCopyOnRead�Read_33/ReadVariableOp�Read_34/DisableCopyOnRead�Read_34/ReadVariableOp�Read_35/DisableCopyOnRead�Read_35/ReadVariableOp�Read_36/DisableCopyOnRead�Read_36/ReadVariableOp�Read_37/DisableCopyOnRead�Read_37/ReadVariableOp�Read_38/DisableCopyOnRead�Read_38/ReadVariableOp�Read_39/DisableCopyOnRead�Read_39/ReadVariableOp�Read_4/DisableCopyOnRead�Read_4/ReadVariableOp�Read_40/DisableCopyOnRead�Read_40/ReadVariableOp�Read_41/DisableCopyOnRead�Read_41/ReadVariableOp�Read_42/DisableCopyOnRead�Read_42/ReadVariableOp�Read_43/DisableCopyOnRead�Read_43/ReadVariableOp�Read_44/DisableCopyOnRead�Read_44/ReadVariableOp�Read_45/DisableCopyOnRead�Read_45/ReadVariableOp�Read_46/DisableCopyOnRead�Read_46/ReadVariableOp�Read_47/DisableCopyOnRead�Read_47/ReadVariableOp�Read_48/DisableCopyOnRead�Read_48/ReadVariableOp�Read_49/DisableCopyOnRead�Read_49/ReadVariableOp�Read_5/DisableCopyOnRead�Read_5/ReadVariableOp�Read_50/DisableCopyOnRead�Read_50/ReadVariableOp�Read_51/DisableCopyOnRead�Read_51/ReadVariableOp�Read_52/DisableCopyOnRead�Read_52/ReadVariableOp�Read_53/DisableCopyOnRead�Read_53/ReadVariableOp�Read_54/DisableCopyOnRead�Read_54/ReadVariableOp�Read_55/DisableCopyOnRead�Read_55/ReadVariableOp�Read_56/DisableCopyOnRead�Read_56/ReadVariableOp�Read_57/DisableCopyOnRead�Read_57/ReadVariableOp�Read_6/DisableCopyOnRead�Read_6/ReadVariableOp�Read_7/DisableCopyOnRead�Read_7/ReadVariableOp�Read_8/DisableCopyOnRead�Read_8/ReadVariableOp�Read_9/DisableCopyOnRead�Read_9/ReadVariableOpw
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
Read_2/DisableCopyOnReadDisableCopyOnRead<read_2_disablecopyonread_fullyconnectedlayerarearatio_kernel"/device:CPU:0*
_output_shapes
 �
Read_2/ReadVariableOpReadVariableOp<read_2_disablecopyonread_fullyconnectedlayerarearatio_kernel^Read_2/DisableCopyOnRead"/device:CPU:0*
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
Read_3/DisableCopyOnReadDisableCopyOnRead:read_3_disablecopyonread_fullyconnectedlayerarearatio_bias"/device:CPU:0*
_output_shapes
 �
Read_3/ReadVariableOpReadVariableOp:read_3_disablecopyonread_fullyconnectedlayerarearatio_bias^Read_3/DisableCopyOnRead"/device:CPU:0*
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
Read_4/DisableCopyOnReadDisableCopyOnRead9read_4_disablecopyonread_fullyconnectedlayersolved_kernel"/device:CPU:0*
_output_shapes
 �
Read_4/ReadVariableOpReadVariableOp9read_4_disablecopyonread_fullyconnectedlayersolved_kernel^Read_4/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:

*
dtype0m

Identity_8IdentityRead_4/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:

c

Identity_9IdentityIdentity_8:output:0"/device:CPU:0*
T0*
_output_shapes

:

�
Read_5/DisableCopyOnReadDisableCopyOnRead7read_5_disablecopyonread_fullyconnectedlayersolved_bias"/device:CPU:0*
_output_shapes
 �
Read_5/ReadVariableOpReadVariableOp7read_5_disablecopyonread_fullyconnectedlayersolved_bias^Read_5/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:
*
dtype0j
Identity_10IdentityRead_5/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:
a
Identity_11IdentityIdentity_10:output:0"/device:CPU:0*
T0*
_output_shapes
:
�
Read_6/DisableCopyOnReadDisableCopyOnRead3read_6_disablecopyonread_predictionarearatio_kernel"/device:CPU:0*
_output_shapes
 �
Read_6/ReadVariableOpReadVariableOp3read_6_disablecopyonread_predictionarearatio_kernel^Read_6/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:
*
dtype0n
Identity_12IdentityRead_6/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:
e
Identity_13IdentityIdentity_12:output:0"/device:CPU:0*
T0*
_output_shapes

:
�
Read_7/DisableCopyOnReadDisableCopyOnRead1read_7_disablecopyonread_predictionarearatio_bias"/device:CPU:0*
_output_shapes
 �
Read_7/ReadVariableOpReadVariableOp1read_7_disablecopyonread_predictionarearatio_bias^Read_7/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0j
Identity_14IdentityRead_7/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_15IdentityIdentity_14:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_8/DisableCopyOnReadDisableCopyOnRead0read_8_disablecopyonread_predictionsolved_kernel"/device:CPU:0*
_output_shapes
 �
Read_8/ReadVariableOpReadVariableOp0read_8_disablecopyonread_predictionsolved_kernel^Read_8/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:
*
dtype0n
Identity_16IdentityRead_8/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:
e
Identity_17IdentityIdentity_16:output:0"/device:CPU:0*
T0*
_output_shapes

:
�
Read_9/DisableCopyOnReadDisableCopyOnRead.read_9_disablecopyonread_predictionsolved_bias"/device:CPU:0*
_output_shapes
 �
Read_9/ReadVariableOpReadVariableOp.read_9_disablecopyonread_predictionsolved_bias^Read_9/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0j
Identity_18IdentityRead_9/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_19IdentityIdentity_18:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_10/DisableCopyOnReadDisableCopyOnReadZread_10_disablecopyonread_transformer_encoder_15_encoder_selfattentionlayer_1_query_kernel"/device:CPU:0*
_output_shapes
 �
Read_10/ReadVariableOpReadVariableOpZread_10_disablecopyonread_transformer_encoder_15_encoder_selfattentionlayer_1_query_kernel^Read_10/DisableCopyOnRead"/device:CPU:0*"
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
Read_11/DisableCopyOnReadDisableCopyOnReadXread_11_disablecopyonread_transformer_encoder_15_encoder_selfattentionlayer_1_query_bias"/device:CPU:0*
_output_shapes
 �
Read_11/ReadVariableOpReadVariableOpXread_11_disablecopyonread_transformer_encoder_15_encoder_selfattentionlayer_1_query_bias^Read_11/DisableCopyOnRead"/device:CPU:0*
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
Read_12/DisableCopyOnReadDisableCopyOnReadXread_12_disablecopyonread_transformer_encoder_15_encoder_selfattentionlayer_1_key_kernel"/device:CPU:0*
_output_shapes
 �
Read_12/ReadVariableOpReadVariableOpXread_12_disablecopyonread_transformer_encoder_15_encoder_selfattentionlayer_1_key_kernel^Read_12/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
:	*
dtype0s
Identity_24IdentityRead_12/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
:	i
Identity_25IdentityIdentity_24:output:0"/device:CPU:0*
T0*"
_output_shapes
:	�
Read_13/DisableCopyOnReadDisableCopyOnReadVread_13_disablecopyonread_transformer_encoder_15_encoder_selfattentionlayer_1_key_bias"/device:CPU:0*
_output_shapes
 �
Read_13/ReadVariableOpReadVariableOpVread_13_disablecopyonread_transformer_encoder_15_encoder_selfattentionlayer_1_key_bias^Read_13/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0o
Identity_26IdentityRead_13/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:e
Identity_27IdentityIdentity_26:output:0"/device:CPU:0*
T0*
_output_shapes

:�
Read_14/DisableCopyOnReadDisableCopyOnReadZread_14_disablecopyonread_transformer_encoder_15_encoder_selfattentionlayer_1_value_kernel"/device:CPU:0*
_output_shapes
 �
Read_14/ReadVariableOpReadVariableOpZread_14_disablecopyonread_transformer_encoder_15_encoder_selfattentionlayer_1_value_kernel^Read_14/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
:	*
dtype0s
Identity_28IdentityRead_14/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
:	i
Identity_29IdentityIdentity_28:output:0"/device:CPU:0*
T0*"
_output_shapes
:	�
Read_15/DisableCopyOnReadDisableCopyOnReadXread_15_disablecopyonread_transformer_encoder_15_encoder_selfattentionlayer_1_value_bias"/device:CPU:0*
_output_shapes
 �
Read_15/ReadVariableOpReadVariableOpXread_15_disablecopyonread_transformer_encoder_15_encoder_selfattentionlayer_1_value_bias^Read_15/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0o
Identity_30IdentityRead_15/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:e
Identity_31IdentityIdentity_30:output:0"/device:CPU:0*
T0*
_output_shapes

:�
Read_16/DisableCopyOnReadDisableCopyOnReaderead_16_disablecopyonread_transformer_encoder_15_encoder_selfattentionlayer_1_attention_output_kernel"/device:CPU:0*
_output_shapes
 �
Read_16/ReadVariableOpReadVariableOperead_16_disablecopyonread_transformer_encoder_15_encoder_selfattentionlayer_1_attention_output_kernel^Read_16/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
:	*
dtype0s
Identity_32IdentityRead_16/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
:	i
Identity_33IdentityIdentity_32:output:0"/device:CPU:0*
T0*"
_output_shapes
:	�
Read_17/DisableCopyOnReadDisableCopyOnReadcread_17_disablecopyonread_transformer_encoder_15_encoder_selfattentionlayer_1_attention_output_bias"/device:CPU:0*
_output_shapes
 �
Read_17/ReadVariableOpReadVariableOpcread_17_disablecopyonread_transformer_encoder_15_encoder_selfattentionlayer_1_attention_output_bias^Read_17/DisableCopyOnRead"/device:CPU:0*
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
Read_18/DisableCopyOnReadDisableCopyOnReadWread_18_disablecopyonread_transformer_encoder_15_encoder_1st_normalizationlayer_1_gamma"/device:CPU:0*
_output_shapes
 �
Read_18/ReadVariableOpReadVariableOpWread_18_disablecopyonread_transformer_encoder_15_encoder_1st_normalizationlayer_1_gamma^Read_18/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	*
dtype0k
Identity_36IdentityRead_18/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	a
Identity_37IdentityIdentity_36:output:0"/device:CPU:0*
T0*
_output_shapes
:	�
Read_19/DisableCopyOnReadDisableCopyOnReadVread_19_disablecopyonread_transformer_encoder_15_encoder_1st_normalizationlayer_1_beta"/device:CPU:0*
_output_shapes
 �
Read_19/ReadVariableOpReadVariableOpVread_19_disablecopyonread_transformer_encoder_15_encoder_1st_normalizationlayer_1_beta^Read_19/DisableCopyOnRead"/device:CPU:0*
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
Read_20/DisableCopyOnReadDisableCopyOnReadWread_20_disablecopyonread_transformer_encoder_15_encoder_2nd_normalizationlayer_1_gamma"/device:CPU:0*
_output_shapes
 �
Read_20/ReadVariableOpReadVariableOpWread_20_disablecopyonread_transformer_encoder_15_encoder_2nd_normalizationlayer_1_gamma^Read_20/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	*
dtype0k
Identity_40IdentityRead_20/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	a
Identity_41IdentityIdentity_40:output:0"/device:CPU:0*
T0*
_output_shapes
:	�
Read_21/DisableCopyOnReadDisableCopyOnReadVread_21_disablecopyonread_transformer_encoder_15_encoder_2nd_normalizationlayer_1_beta"/device:CPU:0*
_output_shapes
 �
Read_21/ReadVariableOpReadVariableOpVread_21_disablecopyonread_transformer_encoder_15_encoder_2nd_normalizationlayer_1_beta^Read_21/DisableCopyOnRead"/device:CPU:0*
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
Read_22/DisableCopyOnReadDisableCopyOnReadTread_22_disablecopyonread_transformer_encoder_15_encoder_feedforwardlayer_1_1_kernel"/device:CPU:0*
_output_shapes
 �
Read_22/ReadVariableOpReadVariableOpTread_22_disablecopyonread_transformer_encoder_15_encoder_feedforwardlayer_1_1_kernel^Read_22/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:		*
dtype0o
Identity_44IdentityRead_22/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:		e
Identity_45IdentityIdentity_44:output:0"/device:CPU:0*
T0*
_output_shapes

:		�
Read_23/DisableCopyOnReadDisableCopyOnReadRread_23_disablecopyonread_transformer_encoder_15_encoder_feedforwardlayer_1_1_bias"/device:CPU:0*
_output_shapes
 �
Read_23/ReadVariableOpReadVariableOpRread_23_disablecopyonread_transformer_encoder_15_encoder_feedforwardlayer_1_1_bias^Read_23/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	*
dtype0k
Identity_46IdentityRead_23/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	a
Identity_47IdentityIdentity_46:output:0"/device:CPU:0*
T0*
_output_shapes
:	�
Read_24/DisableCopyOnReadDisableCopyOnReadTread_24_disablecopyonread_transformer_encoder_15_encoder_feedforwardlayer_2_1_kernel"/device:CPU:0*
_output_shapes
 �
Read_24/ReadVariableOpReadVariableOpTread_24_disablecopyonread_transformer_encoder_15_encoder_feedforwardlayer_2_1_kernel^Read_24/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:		*
dtype0o
Identity_48IdentityRead_24/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:		e
Identity_49IdentityIdentity_48:output:0"/device:CPU:0*
T0*
_output_shapes

:		�
Read_25/DisableCopyOnReadDisableCopyOnReadRread_25_disablecopyonread_transformer_encoder_15_encoder_feedforwardlayer_2_1_bias"/device:CPU:0*
_output_shapes
 �
Read_25/ReadVariableOpReadVariableOpRread_25_disablecopyonread_transformer_encoder_15_encoder_feedforwardlayer_2_1_bias^Read_25/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	*
dtype0k
Identity_50IdentityRead_25/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	a
Identity_51IdentityIdentity_50:output:0"/device:CPU:0*
T0*
_output_shapes
:	�
Read_26/DisableCopyOnReadDisableCopyOnReadZread_26_disablecopyonread_transformer_encoder_16_encoder_selfattentionlayer_2_query_kernel"/device:CPU:0*
_output_shapes
 �
Read_26/ReadVariableOpReadVariableOpZread_26_disablecopyonread_transformer_encoder_16_encoder_selfattentionlayer_2_query_kernel^Read_26/DisableCopyOnRead"/device:CPU:0*"
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
Read_27/DisableCopyOnReadDisableCopyOnReadXread_27_disablecopyonread_transformer_encoder_16_encoder_selfattentionlayer_2_query_bias"/device:CPU:0*
_output_shapes
 �
Read_27/ReadVariableOpReadVariableOpXread_27_disablecopyonread_transformer_encoder_16_encoder_selfattentionlayer_2_query_bias^Read_27/DisableCopyOnRead"/device:CPU:0*
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
Read_28/DisableCopyOnReadDisableCopyOnReadXread_28_disablecopyonread_transformer_encoder_16_encoder_selfattentionlayer_2_key_kernel"/device:CPU:0*
_output_shapes
 �
Read_28/ReadVariableOpReadVariableOpXread_28_disablecopyonread_transformer_encoder_16_encoder_selfattentionlayer_2_key_kernel^Read_28/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
:	*
dtype0s
Identity_56IdentityRead_28/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
:	i
Identity_57IdentityIdentity_56:output:0"/device:CPU:0*
T0*"
_output_shapes
:	�
Read_29/DisableCopyOnReadDisableCopyOnReadVread_29_disablecopyonread_transformer_encoder_16_encoder_selfattentionlayer_2_key_bias"/device:CPU:0*
_output_shapes
 �
Read_29/ReadVariableOpReadVariableOpVread_29_disablecopyonread_transformer_encoder_16_encoder_selfattentionlayer_2_key_bias^Read_29/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0o
Identity_58IdentityRead_29/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:e
Identity_59IdentityIdentity_58:output:0"/device:CPU:0*
T0*
_output_shapes

:�
Read_30/DisableCopyOnReadDisableCopyOnReadZread_30_disablecopyonread_transformer_encoder_16_encoder_selfattentionlayer_2_value_kernel"/device:CPU:0*
_output_shapes
 �
Read_30/ReadVariableOpReadVariableOpZread_30_disablecopyonread_transformer_encoder_16_encoder_selfattentionlayer_2_value_kernel^Read_30/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
:	*
dtype0s
Identity_60IdentityRead_30/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
:	i
Identity_61IdentityIdentity_60:output:0"/device:CPU:0*
T0*"
_output_shapes
:	�
Read_31/DisableCopyOnReadDisableCopyOnReadXread_31_disablecopyonread_transformer_encoder_16_encoder_selfattentionlayer_2_value_bias"/device:CPU:0*
_output_shapes
 �
Read_31/ReadVariableOpReadVariableOpXread_31_disablecopyonread_transformer_encoder_16_encoder_selfattentionlayer_2_value_bias^Read_31/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0o
Identity_62IdentityRead_31/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:e
Identity_63IdentityIdentity_62:output:0"/device:CPU:0*
T0*
_output_shapes

:�
Read_32/DisableCopyOnReadDisableCopyOnReaderead_32_disablecopyonread_transformer_encoder_16_encoder_selfattentionlayer_2_attention_output_kernel"/device:CPU:0*
_output_shapes
 �
Read_32/ReadVariableOpReadVariableOperead_32_disablecopyonread_transformer_encoder_16_encoder_selfattentionlayer_2_attention_output_kernel^Read_32/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
:	*
dtype0s
Identity_64IdentityRead_32/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
:	i
Identity_65IdentityIdentity_64:output:0"/device:CPU:0*
T0*"
_output_shapes
:	�
Read_33/DisableCopyOnReadDisableCopyOnReadcread_33_disablecopyonread_transformer_encoder_16_encoder_selfattentionlayer_2_attention_output_bias"/device:CPU:0*
_output_shapes
 �
Read_33/ReadVariableOpReadVariableOpcread_33_disablecopyonread_transformer_encoder_16_encoder_selfattentionlayer_2_attention_output_bias^Read_33/DisableCopyOnRead"/device:CPU:0*
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
Read_34/DisableCopyOnReadDisableCopyOnReadWread_34_disablecopyonread_transformer_encoder_16_encoder_1st_normalizationlayer_2_gamma"/device:CPU:0*
_output_shapes
 �
Read_34/ReadVariableOpReadVariableOpWread_34_disablecopyonread_transformer_encoder_16_encoder_1st_normalizationlayer_2_gamma^Read_34/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	*
dtype0k
Identity_68IdentityRead_34/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	a
Identity_69IdentityIdentity_68:output:0"/device:CPU:0*
T0*
_output_shapes
:	�
Read_35/DisableCopyOnReadDisableCopyOnReadVread_35_disablecopyonread_transformer_encoder_16_encoder_1st_normalizationlayer_2_beta"/device:CPU:0*
_output_shapes
 �
Read_35/ReadVariableOpReadVariableOpVread_35_disablecopyonread_transformer_encoder_16_encoder_1st_normalizationlayer_2_beta^Read_35/DisableCopyOnRead"/device:CPU:0*
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
Read_36/DisableCopyOnReadDisableCopyOnReadWread_36_disablecopyonread_transformer_encoder_16_encoder_2nd_normalizationlayer_2_gamma"/device:CPU:0*
_output_shapes
 �
Read_36/ReadVariableOpReadVariableOpWread_36_disablecopyonread_transformer_encoder_16_encoder_2nd_normalizationlayer_2_gamma^Read_36/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	*
dtype0k
Identity_72IdentityRead_36/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	a
Identity_73IdentityIdentity_72:output:0"/device:CPU:0*
T0*
_output_shapes
:	�
Read_37/DisableCopyOnReadDisableCopyOnReadVread_37_disablecopyonread_transformer_encoder_16_encoder_2nd_normalizationlayer_2_beta"/device:CPU:0*
_output_shapes
 �
Read_37/ReadVariableOpReadVariableOpVread_37_disablecopyonread_transformer_encoder_16_encoder_2nd_normalizationlayer_2_beta^Read_37/DisableCopyOnRead"/device:CPU:0*
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
Read_38/DisableCopyOnReadDisableCopyOnReadTread_38_disablecopyonread_transformer_encoder_16_encoder_feedforwardlayer_1_2_kernel"/device:CPU:0*
_output_shapes
 �
Read_38/ReadVariableOpReadVariableOpTread_38_disablecopyonread_transformer_encoder_16_encoder_feedforwardlayer_1_2_kernel^Read_38/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:		*
dtype0o
Identity_76IdentityRead_38/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:		e
Identity_77IdentityIdentity_76:output:0"/device:CPU:0*
T0*
_output_shapes

:		�
Read_39/DisableCopyOnReadDisableCopyOnReadRread_39_disablecopyonread_transformer_encoder_16_encoder_feedforwardlayer_1_2_bias"/device:CPU:0*
_output_shapes
 �
Read_39/ReadVariableOpReadVariableOpRread_39_disablecopyonread_transformer_encoder_16_encoder_feedforwardlayer_1_2_bias^Read_39/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	*
dtype0k
Identity_78IdentityRead_39/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	a
Identity_79IdentityIdentity_78:output:0"/device:CPU:0*
T0*
_output_shapes
:	�
Read_40/DisableCopyOnReadDisableCopyOnReadTread_40_disablecopyonread_transformer_encoder_16_encoder_feedforwardlayer_2_2_kernel"/device:CPU:0*
_output_shapes
 �
Read_40/ReadVariableOpReadVariableOpTread_40_disablecopyonread_transformer_encoder_16_encoder_feedforwardlayer_2_2_kernel^Read_40/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:		*
dtype0o
Identity_80IdentityRead_40/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:		e
Identity_81IdentityIdentity_80:output:0"/device:CPU:0*
T0*
_output_shapes

:		�
Read_41/DisableCopyOnReadDisableCopyOnReadRread_41_disablecopyonread_transformer_encoder_16_encoder_feedforwardlayer_2_2_bias"/device:CPU:0*
_output_shapes
 �
Read_41/ReadVariableOpReadVariableOpRread_41_disablecopyonread_transformer_encoder_16_encoder_feedforwardlayer_2_2_bias^Read_41/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	*
dtype0k
Identity_82IdentityRead_41/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	a
Identity_83IdentityIdentity_82:output:0"/device:CPU:0*
T0*
_output_shapes
:	�
Read_42/DisableCopyOnReadDisableCopyOnReadZread_42_disablecopyonread_transformer_encoder_17_encoder_selfattentionlayer_3_query_kernel"/device:CPU:0*
_output_shapes
 �
Read_42/ReadVariableOpReadVariableOpZread_42_disablecopyonread_transformer_encoder_17_encoder_selfattentionlayer_3_query_kernel^Read_42/DisableCopyOnRead"/device:CPU:0*"
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
Read_43/DisableCopyOnReadDisableCopyOnReadXread_43_disablecopyonread_transformer_encoder_17_encoder_selfattentionlayer_3_query_bias"/device:CPU:0*
_output_shapes
 �
Read_43/ReadVariableOpReadVariableOpXread_43_disablecopyonread_transformer_encoder_17_encoder_selfattentionlayer_3_query_bias^Read_43/DisableCopyOnRead"/device:CPU:0*
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
Read_44/DisableCopyOnReadDisableCopyOnReadXread_44_disablecopyonread_transformer_encoder_17_encoder_selfattentionlayer_3_key_kernel"/device:CPU:0*
_output_shapes
 �
Read_44/ReadVariableOpReadVariableOpXread_44_disablecopyonread_transformer_encoder_17_encoder_selfattentionlayer_3_key_kernel^Read_44/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
:	*
dtype0s
Identity_88IdentityRead_44/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
:	i
Identity_89IdentityIdentity_88:output:0"/device:CPU:0*
T0*"
_output_shapes
:	�
Read_45/DisableCopyOnReadDisableCopyOnReadVread_45_disablecopyonread_transformer_encoder_17_encoder_selfattentionlayer_3_key_bias"/device:CPU:0*
_output_shapes
 �
Read_45/ReadVariableOpReadVariableOpVread_45_disablecopyonread_transformer_encoder_17_encoder_selfattentionlayer_3_key_bias^Read_45/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0o
Identity_90IdentityRead_45/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:e
Identity_91IdentityIdentity_90:output:0"/device:CPU:0*
T0*
_output_shapes

:�
Read_46/DisableCopyOnReadDisableCopyOnReadZread_46_disablecopyonread_transformer_encoder_17_encoder_selfattentionlayer_3_value_kernel"/device:CPU:0*
_output_shapes
 �
Read_46/ReadVariableOpReadVariableOpZread_46_disablecopyonread_transformer_encoder_17_encoder_selfattentionlayer_3_value_kernel^Read_46/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
:	*
dtype0s
Identity_92IdentityRead_46/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
:	i
Identity_93IdentityIdentity_92:output:0"/device:CPU:0*
T0*"
_output_shapes
:	�
Read_47/DisableCopyOnReadDisableCopyOnReadXread_47_disablecopyonread_transformer_encoder_17_encoder_selfattentionlayer_3_value_bias"/device:CPU:0*
_output_shapes
 �
Read_47/ReadVariableOpReadVariableOpXread_47_disablecopyonread_transformer_encoder_17_encoder_selfattentionlayer_3_value_bias^Read_47/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0o
Identity_94IdentityRead_47/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:e
Identity_95IdentityIdentity_94:output:0"/device:CPU:0*
T0*
_output_shapes

:�
Read_48/DisableCopyOnReadDisableCopyOnReaderead_48_disablecopyonread_transformer_encoder_17_encoder_selfattentionlayer_3_attention_output_kernel"/device:CPU:0*
_output_shapes
 �
Read_48/ReadVariableOpReadVariableOperead_48_disablecopyonread_transformer_encoder_17_encoder_selfattentionlayer_3_attention_output_kernel^Read_48/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
:	*
dtype0s
Identity_96IdentityRead_48/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
:	i
Identity_97IdentityIdentity_96:output:0"/device:CPU:0*
T0*"
_output_shapes
:	�
Read_49/DisableCopyOnReadDisableCopyOnReadcread_49_disablecopyonread_transformer_encoder_17_encoder_selfattentionlayer_3_attention_output_bias"/device:CPU:0*
_output_shapes
 �
Read_49/ReadVariableOpReadVariableOpcread_49_disablecopyonread_transformer_encoder_17_encoder_selfattentionlayer_3_attention_output_bias^Read_49/DisableCopyOnRead"/device:CPU:0*
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
Read_50/DisableCopyOnReadDisableCopyOnReadWread_50_disablecopyonread_transformer_encoder_17_encoder_1st_normalizationlayer_3_gamma"/device:CPU:0*
_output_shapes
 �
Read_50/ReadVariableOpReadVariableOpWread_50_disablecopyonread_transformer_encoder_17_encoder_1st_normalizationlayer_3_gamma^Read_50/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	*
dtype0l
Identity_100IdentityRead_50/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	c
Identity_101IdentityIdentity_100:output:0"/device:CPU:0*
T0*
_output_shapes
:	�
Read_51/DisableCopyOnReadDisableCopyOnReadVread_51_disablecopyonread_transformer_encoder_17_encoder_1st_normalizationlayer_3_beta"/device:CPU:0*
_output_shapes
 �
Read_51/ReadVariableOpReadVariableOpVread_51_disablecopyonread_transformer_encoder_17_encoder_1st_normalizationlayer_3_beta^Read_51/DisableCopyOnRead"/device:CPU:0*
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
Read_52/DisableCopyOnReadDisableCopyOnReadWread_52_disablecopyonread_transformer_encoder_17_encoder_2nd_normalizationlayer_3_gamma"/device:CPU:0*
_output_shapes
 �
Read_52/ReadVariableOpReadVariableOpWread_52_disablecopyonread_transformer_encoder_17_encoder_2nd_normalizationlayer_3_gamma^Read_52/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	*
dtype0l
Identity_104IdentityRead_52/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	c
Identity_105IdentityIdentity_104:output:0"/device:CPU:0*
T0*
_output_shapes
:	�
Read_53/DisableCopyOnReadDisableCopyOnReadVread_53_disablecopyonread_transformer_encoder_17_encoder_2nd_normalizationlayer_3_beta"/device:CPU:0*
_output_shapes
 �
Read_53/ReadVariableOpReadVariableOpVread_53_disablecopyonread_transformer_encoder_17_encoder_2nd_normalizationlayer_3_beta^Read_53/DisableCopyOnRead"/device:CPU:0*
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
:	�
Read_54/DisableCopyOnReadDisableCopyOnReadTread_54_disablecopyonread_transformer_encoder_17_encoder_feedforwardlayer_1_3_kernel"/device:CPU:0*
_output_shapes
 �
Read_54/ReadVariableOpReadVariableOpTread_54_disablecopyonread_transformer_encoder_17_encoder_feedforwardlayer_1_3_kernel^Read_54/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:		*
dtype0p
Identity_108IdentityRead_54/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:		g
Identity_109IdentityIdentity_108:output:0"/device:CPU:0*
T0*
_output_shapes

:		�
Read_55/DisableCopyOnReadDisableCopyOnReadRread_55_disablecopyonread_transformer_encoder_17_encoder_feedforwardlayer_1_3_bias"/device:CPU:0*
_output_shapes
 �
Read_55/ReadVariableOpReadVariableOpRread_55_disablecopyonread_transformer_encoder_17_encoder_feedforwardlayer_1_3_bias^Read_55/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	*
dtype0l
Identity_110IdentityRead_55/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	c
Identity_111IdentityIdentity_110:output:0"/device:CPU:0*
T0*
_output_shapes
:	�
Read_56/DisableCopyOnReadDisableCopyOnReadTread_56_disablecopyonread_transformer_encoder_17_encoder_feedforwardlayer_2_3_kernel"/device:CPU:0*
_output_shapes
 �
Read_56/ReadVariableOpReadVariableOpTread_56_disablecopyonread_transformer_encoder_17_encoder_feedforwardlayer_2_3_kernel^Read_56/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:		*
dtype0p
Identity_112IdentityRead_56/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:		g
Identity_113IdentityIdentity_112:output:0"/device:CPU:0*
T0*
_output_shapes

:		�
Read_57/DisableCopyOnReadDisableCopyOnReadRread_57_disablecopyonread_transformer_encoder_17_encoder_feedforwardlayer_2_3_bias"/device:CPU:0*
_output_shapes
 �
Read_57/ReadVariableOpReadVariableOpRread_57_disablecopyonread_transformer_encoder_17_encoder_feedforwardlayer_2_3_bias^Read_57/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	*
dtype0l
Identity_114IdentityRead_57/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	c
Identity_115IdentityIdentity_114:output:0"/device:CPU:0*
T0*
_output_shapes
:	�
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:;*
dtype0*�
value�B�;B5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB'variables/26/.ATTRIBUTES/VARIABLE_VALUEB'variables/27/.ATTRIBUTES/VARIABLE_VALUEB'variables/28/.ATTRIBUTES/VARIABLE_VALUEB'variables/29/.ATTRIBUTES/VARIABLE_VALUEB'variables/30/.ATTRIBUTES/VARIABLE_VALUEB'variables/31/.ATTRIBUTES/VARIABLE_VALUEB'variables/32/.ATTRIBUTES/VARIABLE_VALUEB'variables/33/.ATTRIBUTES/VARIABLE_VALUEB'variables/34/.ATTRIBUTES/VARIABLE_VALUEB'variables/35/.ATTRIBUTES/VARIABLE_VALUEB'variables/36/.ATTRIBUTES/VARIABLE_VALUEB'variables/37/.ATTRIBUTES/VARIABLE_VALUEB'variables/38/.ATTRIBUTES/VARIABLE_VALUEB'variables/39/.ATTRIBUTES/VARIABLE_VALUEB'variables/40/.ATTRIBUTES/VARIABLE_VALUEB'variables/41/.ATTRIBUTES/VARIABLE_VALUEB'variables/42/.ATTRIBUTES/VARIABLE_VALUEB'variables/43/.ATTRIBUTES/VARIABLE_VALUEB'variables/44/.ATTRIBUTES/VARIABLE_VALUEB'variables/45/.ATTRIBUTES/VARIABLE_VALUEB'variables/46/.ATTRIBUTES/VARIABLE_VALUEB'variables/47/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:;*
dtype0*�
value�B~;B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Identity_1:output:0Identity_3:output:0Identity_5:output:0Identity_7:output:0Identity_9:output:0Identity_11:output:0Identity_13:output:0Identity_15:output:0Identity_17:output:0Identity_19:output:0Identity_21:output:0Identity_23:output:0Identity_25:output:0Identity_27:output:0Identity_29:output:0Identity_31:output:0Identity_33:output:0Identity_35:output:0Identity_37:output:0Identity_39:output:0Identity_41:output:0Identity_43:output:0Identity_45:output:0Identity_47:output:0Identity_49:output:0Identity_51:output:0Identity_53:output:0Identity_55:output:0Identity_57:output:0Identity_59:output:0Identity_61:output:0Identity_63:output:0Identity_65:output:0Identity_67:output:0Identity_69:output:0Identity_71:output:0Identity_73:output:0Identity_75:output:0Identity_77:output:0Identity_79:output:0Identity_81:output:0Identity_83:output:0Identity_85:output:0Identity_87:output:0Identity_89:output:0Identity_91:output:0Identity_93:output:0Identity_95:output:0Identity_97:output:0Identity_99:output:0Identity_101:output:0Identity_103:output:0Identity_105:output:0Identity_107:output:0Identity_109:output:0Identity_111:output:0Identity_113:output:0Identity_115:output:0savev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *I
dtypes?
=2;�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 j
Identity_116Identityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: W
Identity_117IdentityIdentity_116:output:0^NoOp*
T0*
_output_shapes
: �
NoOpNoOp^MergeV2Checkpoints^Read/DisableCopyOnRead^Read/ReadVariableOp^Read_1/DisableCopyOnRead^Read_1/ReadVariableOp^Read_10/DisableCopyOnRead^Read_10/ReadVariableOp^Read_11/DisableCopyOnRead^Read_11/ReadVariableOp^Read_12/DisableCopyOnRead^Read_12/ReadVariableOp^Read_13/DisableCopyOnRead^Read_13/ReadVariableOp^Read_14/DisableCopyOnRead^Read_14/ReadVariableOp^Read_15/DisableCopyOnRead^Read_15/ReadVariableOp^Read_16/DisableCopyOnRead^Read_16/ReadVariableOp^Read_17/DisableCopyOnRead^Read_17/ReadVariableOp^Read_18/DisableCopyOnRead^Read_18/ReadVariableOp^Read_19/DisableCopyOnRead^Read_19/ReadVariableOp^Read_2/DisableCopyOnRead^Read_2/ReadVariableOp^Read_20/DisableCopyOnRead^Read_20/ReadVariableOp^Read_21/DisableCopyOnRead^Read_21/ReadVariableOp^Read_22/DisableCopyOnRead^Read_22/ReadVariableOp^Read_23/DisableCopyOnRead^Read_23/ReadVariableOp^Read_24/DisableCopyOnRead^Read_24/ReadVariableOp^Read_25/DisableCopyOnRead^Read_25/ReadVariableOp^Read_26/DisableCopyOnRead^Read_26/ReadVariableOp^Read_27/DisableCopyOnRead^Read_27/ReadVariableOp^Read_28/DisableCopyOnRead^Read_28/ReadVariableOp^Read_29/DisableCopyOnRead^Read_29/ReadVariableOp^Read_3/DisableCopyOnRead^Read_3/ReadVariableOp^Read_30/DisableCopyOnRead^Read_30/ReadVariableOp^Read_31/DisableCopyOnRead^Read_31/ReadVariableOp^Read_32/DisableCopyOnRead^Read_32/ReadVariableOp^Read_33/DisableCopyOnRead^Read_33/ReadVariableOp^Read_34/DisableCopyOnRead^Read_34/ReadVariableOp^Read_35/DisableCopyOnRead^Read_35/ReadVariableOp^Read_36/DisableCopyOnRead^Read_36/ReadVariableOp^Read_37/DisableCopyOnRead^Read_37/ReadVariableOp^Read_38/DisableCopyOnRead^Read_38/ReadVariableOp^Read_39/DisableCopyOnRead^Read_39/ReadVariableOp^Read_4/DisableCopyOnRead^Read_4/ReadVariableOp^Read_40/DisableCopyOnRead^Read_40/ReadVariableOp^Read_41/DisableCopyOnRead^Read_41/ReadVariableOp^Read_42/DisableCopyOnRead^Read_42/ReadVariableOp^Read_43/DisableCopyOnRead^Read_43/ReadVariableOp^Read_44/DisableCopyOnRead^Read_44/ReadVariableOp^Read_45/DisableCopyOnRead^Read_45/ReadVariableOp^Read_46/DisableCopyOnRead^Read_46/ReadVariableOp^Read_47/DisableCopyOnRead^Read_47/ReadVariableOp^Read_48/DisableCopyOnRead^Read_48/ReadVariableOp^Read_49/DisableCopyOnRead^Read_49/ReadVariableOp^Read_5/DisableCopyOnRead^Read_5/ReadVariableOp^Read_50/DisableCopyOnRead^Read_50/ReadVariableOp^Read_51/DisableCopyOnRead^Read_51/ReadVariableOp^Read_52/DisableCopyOnRead^Read_52/ReadVariableOp^Read_53/DisableCopyOnRead^Read_53/ReadVariableOp^Read_54/DisableCopyOnRead^Read_54/ReadVariableOp^Read_55/DisableCopyOnRead^Read_55/ReadVariableOp^Read_56/DisableCopyOnRead^Read_56/ReadVariableOp^Read_57/DisableCopyOnRead^Read_57/ReadVariableOp^Read_6/DisableCopyOnRead^Read_6/ReadVariableOp^Read_7/DisableCopyOnRead^Read_7/ReadVariableOp^Read_8/DisableCopyOnRead^Read_8/ReadVariableOp^Read_9/DisableCopyOnRead^Read_9/ReadVariableOp*
_output_shapes
 "%
identity_117Identity_117:output:0*(
_construction_contextkEagerRuntime*�
_input_shapesz
x: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2(
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
Read_53/ReadVariableOpRead_53/ReadVariableOp26
Read_54/DisableCopyOnReadRead_54/DisableCopyOnRead20
Read_54/ReadVariableOpRead_54/ReadVariableOp26
Read_55/DisableCopyOnReadRead_55/DisableCopyOnRead20
Read_55/ReadVariableOpRead_55/ReadVariableOp26
Read_56/DisableCopyOnReadRead_56/DisableCopyOnRead20
Read_56/ReadVariableOpRead_56/ReadVariableOp26
Read_57/DisableCopyOnReadRead_57/DisableCopyOnRead20
Read_57/ReadVariableOpRead_57/ReadVariableOp24
Read_6/DisableCopyOnReadRead_6/DisableCopyOnRead2.
Read_6/ReadVariableOpRead_6/ReadVariableOp24
Read_7/DisableCopyOnReadRead_7/DisableCopyOnRead2.
Read_7/ReadVariableOpRead_7/ReadVariableOp24
Read_8/DisableCopyOnReadRead_8/DisableCopyOnRead2.
Read_8/ReadVariableOpRead_8/ReadVariableOp24
Read_9/DisableCopyOnReadRead_9/DisableCopyOnRead2.
Read_9/ReadVariableOpRead_9/ReadVariableOp:=;9

_output_shapes
: 

_user_specified_nameConst:X:T
R
_user_specified_name:8transformer_encoder_17/Encoder-FeedForwardLayer_2_3/bias:Z9V
T
_user_specified_name<:transformer_encoder_17/Encoder-FeedForwardLayer_2_3/kernel:X8T
R
_user_specified_name:8transformer_encoder_17/Encoder-FeedForwardLayer_1_3/bias:Z7V
T
_user_specified_name<:transformer_encoder_17/Encoder-FeedForwardLayer_1_3/kernel:\6X
V
_user_specified_name><transformer_encoder_17/Encoder-2nd-NormalizationLayer-3/beta:]5Y
W
_user_specified_name?=transformer_encoder_17/Encoder-2nd-NormalizationLayer-3/gamma:\4X
V
_user_specified_name><transformer_encoder_17/Encoder-1st-NormalizationLayer-3/beta:]3Y
W
_user_specified_name?=transformer_encoder_17/Encoder-1st-NormalizationLayer-3/gamma:i2e
c
_user_specified_nameKItransformer_encoder_17/Encoder-SelfAttentionLayer-3/attention_output/bias:k1g
e
_user_specified_nameMKtransformer_encoder_17/Encoder-SelfAttentionLayer-3/attention_output/kernel:^0Z
X
_user_specified_name@>transformer_encoder_17/Encoder-SelfAttentionLayer-3/value/bias:`/\
Z
_user_specified_nameB@transformer_encoder_17/Encoder-SelfAttentionLayer-3/value/kernel:\.X
V
_user_specified_name><transformer_encoder_17/Encoder-SelfAttentionLayer-3/key/bias:^-Z
X
_user_specified_name@>transformer_encoder_17/Encoder-SelfAttentionLayer-3/key/kernel:^,Z
X
_user_specified_name@>transformer_encoder_17/Encoder-SelfAttentionLayer-3/query/bias:`+\
Z
_user_specified_nameB@transformer_encoder_17/Encoder-SelfAttentionLayer-3/query/kernel:X*T
R
_user_specified_name:8transformer_encoder_16/Encoder-FeedForwardLayer_2_2/bias:Z)V
T
_user_specified_name<:transformer_encoder_16/Encoder-FeedForwardLayer_2_2/kernel:X(T
R
_user_specified_name:8transformer_encoder_16/Encoder-FeedForwardLayer_1_2/bias:Z'V
T
_user_specified_name<:transformer_encoder_16/Encoder-FeedForwardLayer_1_2/kernel:\&X
V
_user_specified_name><transformer_encoder_16/Encoder-2nd-NormalizationLayer-2/beta:]%Y
W
_user_specified_name?=transformer_encoder_16/Encoder-2nd-NormalizationLayer-2/gamma:\$X
V
_user_specified_name><transformer_encoder_16/Encoder-1st-NormalizationLayer-2/beta:]#Y
W
_user_specified_name?=transformer_encoder_16/Encoder-1st-NormalizationLayer-2/gamma:i"e
c
_user_specified_nameKItransformer_encoder_16/Encoder-SelfAttentionLayer-2/attention_output/bias:k!g
e
_user_specified_nameMKtransformer_encoder_16/Encoder-SelfAttentionLayer-2/attention_output/kernel:^ Z
X
_user_specified_name@>transformer_encoder_16/Encoder-SelfAttentionLayer-2/value/bias:`\
Z
_user_specified_nameB@transformer_encoder_16/Encoder-SelfAttentionLayer-2/value/kernel:\X
V
_user_specified_name><transformer_encoder_16/Encoder-SelfAttentionLayer-2/key/bias:^Z
X
_user_specified_name@>transformer_encoder_16/Encoder-SelfAttentionLayer-2/key/kernel:^Z
X
_user_specified_name@>transformer_encoder_16/Encoder-SelfAttentionLayer-2/query/bias:`\
Z
_user_specified_nameB@transformer_encoder_16/Encoder-SelfAttentionLayer-2/query/kernel:XT
R
_user_specified_name:8transformer_encoder_15/Encoder-FeedForwardLayer_2_1/bias:ZV
T
_user_specified_name<:transformer_encoder_15/Encoder-FeedForwardLayer_2_1/kernel:XT
R
_user_specified_name:8transformer_encoder_15/Encoder-FeedForwardLayer_1_1/bias:ZV
T
_user_specified_name<:transformer_encoder_15/Encoder-FeedForwardLayer_1_1/kernel:\X
V
_user_specified_name><transformer_encoder_15/Encoder-2nd-NormalizationLayer-1/beta:]Y
W
_user_specified_name?=transformer_encoder_15/Encoder-2nd-NormalizationLayer-1/gamma:\X
V
_user_specified_name><transformer_encoder_15/Encoder-1st-NormalizationLayer-1/beta:]Y
W
_user_specified_name?=transformer_encoder_15/Encoder-1st-NormalizationLayer-1/gamma:ie
c
_user_specified_nameKItransformer_encoder_15/Encoder-SelfAttentionLayer-1/attention_output/bias:kg
e
_user_specified_nameMKtransformer_encoder_15/Encoder-SelfAttentionLayer-1/attention_output/kernel:^Z
X
_user_specified_name@>transformer_encoder_15/Encoder-SelfAttentionLayer-1/value/bias:`\
Z
_user_specified_nameB@transformer_encoder_15/Encoder-SelfAttentionLayer-1/value/kernel:\X
V
_user_specified_name><transformer_encoder_15/Encoder-SelfAttentionLayer-1/key/bias:^Z
X
_user_specified_name@>transformer_encoder_15/Encoder-SelfAttentionLayer-1/key/kernel:^Z
X
_user_specified_name@>transformer_encoder_15/Encoder-SelfAttentionLayer-1/query/bias:`\
Z
_user_specified_nameB@transformer_encoder_15/Encoder-SelfAttentionLayer-1/query/kernel:5
1
/
_user_specified_namePredictionSolved/bias:7	3
1
_user_specified_namePredictionSolved/kernel:84
2
_user_specified_namePredictionAreaRatio/bias::6
4
_user_specified_namePredictionAreaRatio/kernel:>:
8
_user_specified_name FullyConnectedLayerSolved/bias:@<
:
_user_specified_name" FullyConnectedLayerSolved/kernel:A=
;
_user_specified_name#!FullyConnectedLayerAreaRatio/bias:C?
=
_user_specified_name%#FullyConnectedLayerAreaRatio/kernel:3/
-
_user_specified_nameFinalLayerNorm/beta:40
.
_user_specified_nameFinalLayerNorm/gamma:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
_
StackLevelInputFeaturesD
)serving_default_StackLevelInputFeatures:0���������P	
I
TimeLimitInput7
 serving_default_TimeLimitInput:0���������-
Output_1!
StatefulPartitionedCall:1+
Output!
StatefulPartitionedCall:0tensorflow/serving/predict:��
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
layer_with_weights-6
layer-12
layer_with_weights-7
layer-13
layer-14
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures"
_tf_keras_network
"
_tf_keras_input_layer
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
�
	variables
trainable_variables
 regularization_losses
!	keras_api
"__call__
*#&call_and_return_all_conditional_losses
$self_attention_layer
%add1
&add2
'
layernorm1
(
layernorm2
)feed_forward_layer_1
*feed_forward_layer_2
+dropout_layer"
_tf_keras_layer
�
,	variables
-trainable_variables
.regularization_losses
/	keras_api
0__call__
*1&call_and_return_all_conditional_losses
2self_attention_layer
3add1
4add2
5
layernorm1
6
layernorm2
7feed_forward_layer_1
8feed_forward_layer_2
9dropout_layer"
_tf_keras_layer
�
:	variables
;trainable_variables
<regularization_losses
=	keras_api
>__call__
*?&call_and_return_all_conditional_losses
@self_attention_layer
Aadd1
Badd2
C
layernorm1
D
layernorm2
Efeed_forward_layer_1
Ffeed_forward_layer_2
Gdropout_layer"
_tf_keras_layer
�
H	variables
Itrainable_variables
Jregularization_losses
K	keras_api
L__call__
*M&call_and_return_all_conditional_losses
Naxis
	Ogamma
Pbeta"
_tf_keras_layer
"
_tf_keras_input_layer
�
Q	variables
Rtrainable_variables
Sregularization_losses
T	keras_api
U__call__
*V&call_and_return_all_conditional_losses"
_tf_keras_layer
�
W	variables
Xtrainable_variables
Yregularization_losses
Z	keras_api
[__call__
*\&call_and_return_all_conditional_losses"
_tf_keras_layer
�
]	variables
^trainable_variables
_regularization_losses
`	keras_api
a__call__
*b&call_and_return_all_conditional_losses"
_tf_keras_layer
�
c	variables
dtrainable_variables
eregularization_losses
f	keras_api
g__call__
*h&call_and_return_all_conditional_losses

ikernel
jbias"
_tf_keras_layer
�
k	variables
ltrainable_variables
mregularization_losses
n	keras_api
o__call__
*p&call_and_return_all_conditional_losses

qkernel
rbias"
_tf_keras_layer
�
s	variables
ttrainable_variables
uregularization_losses
v	keras_api
w__call__
*x&call_and_return_all_conditional_losses

ykernel
zbias"
_tf_keras_layer
�
{	variables
|trainable_variables
}regularization_losses
~	keras_api
__call__
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
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
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
O48
P49
i50
j51
q52
r53
y54
z55
�56
�57"
trackable_list_wrapper
�
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
O48
P49
i50
j51
q52
r53
y54
z55
�56
�57"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
)__inference_model_5_layer_call_fn_1430820
)__inference_model_5_layer_call_fn_1430944�
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
D__inference_model_5_layer_call_and_return_conditional_losses_1430011
D__inference_model_5_layer_call_and_return_conditional_losses_1430696�
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
"__inference__wrapped_model_1429166StackLevelInputFeaturesTimeLimitInput"�
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
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
.__inference_MaskingLayer_layer_call_fn_1431394�
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
I__inference_MaskingLayer_layer_call_and_return_conditional_losses_1431405�
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
	variables
trainable_variables
 regularization_losses
"__call__
*#&call_and_return_all_conditional_losses
&#"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
8__inference_transformer_encoder_15_layer_call_fn_1431444
8__inference_transformer_encoder_15_layer_call_fn_1431483�
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
S__inference_transformer_encoder_15_layer_call_and_return_conditional_losses_1431671
S__inference_transformer_encoder_15_layer_call_and_return_conditional_losses_1431845�
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
,	variables
-trainable_variables
.regularization_losses
0__call__
*1&call_and_return_all_conditional_losses
&1"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
8__inference_transformer_encoder_16_layer_call_fn_1431884
8__inference_transformer_encoder_16_layer_call_fn_1431923�
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
S__inference_transformer_encoder_16_layer_call_and_return_conditional_losses_1432111
S__inference_transformer_encoder_16_layer_call_and_return_conditional_losses_1432285�
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
:	variables
;trainable_variables
<regularization_losses
>__call__
*?&call_and_return_all_conditional_losses
&?"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
8__inference_transformer_encoder_17_layer_call_fn_1432324
8__inference_transformer_encoder_17_layer_call_fn_1432363�
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
S__inference_transformer_encoder_17_layer_call_and_return_conditional_losses_1432551
S__inference_transformer_encoder_17_layer_call_and_return_conditional_losses_1432725�
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
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias"
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
.
O0
P1"
trackable_list_wrapper
.
O0
P1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
H	variables
Itrainable_variables
Jregularization_losses
L__call__
*M&call_and_return_all_conditional_losses
&M"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
0__inference_FinalLayerNorm_layer_call_fn_1432734�
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
K__inference_FinalLayerNorm_layer_call_and_return_conditional_losses_1432776�
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
Q	variables
Rtrainable_variables
Sregularization_losses
U__call__
*V&call_and_return_all_conditional_losses
&V"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
B__inference_ReduceStackDimensionViaSummation_layer_call_fn_1432781
B__inference_ReduceStackDimensionViaSummation_layer_call_fn_1432786�
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
]__inference_ReduceStackDimensionViaSummation_layer_call_and_return_conditional_losses_1432794
]__inference_ReduceStackDimensionViaSummation_layer_call_and_return_conditional_losses_1432802�
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
W	variables
Xtrainable_variables
Yregularization_losses
[__call__
*\&call_and_return_all_conditional_losses
&\"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
6__inference_StandardizeTimeLimit_layer_call_fn_1432807
6__inference_StandardizeTimeLimit_layer_call_fn_1432812�
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
Q__inference_StandardizeTimeLimit_layer_call_and_return_conditional_losses_1432820
Q__inference_StandardizeTimeLimit_layer_call_and_return_conditional_losses_1432828�
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
]	variables
^trainable_variables
_regularization_losses
a__call__
*b&call_and_return_all_conditional_losses
&b"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
2__inference_ConcatenateLayer_layer_call_fn_1432834�
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
M__inference_ConcatenateLayer_layer_call_and_return_conditional_losses_1432841�
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
i0
j1"
trackable_list_wrapper
.
i0
j1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
c	variables
dtrainable_variables
eregularization_losses
g__call__
*h&call_and_return_all_conditional_losses
&h"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
>__inference_FullyConnectedLayerAreaRatio_layer_call_fn_1432850�
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
Y__inference_FullyConnectedLayerAreaRatio_layer_call_and_return_conditional_losses_1432868�
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
5:3

2#FullyConnectedLayerAreaRatio/kernel
/:-
2!FullyConnectedLayerAreaRatio/bias
.
q0
r1"
trackable_list_wrapper
.
q0
r1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
k	variables
ltrainable_variables
mregularization_losses
o__call__
*p&call_and_return_all_conditional_losses
&p"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
;__inference_FullyConnectedLayerSolved_layer_call_fn_1432877�
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
V__inference_FullyConnectedLayerSolved_layer_call_and_return_conditional_losses_1432895�
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
2:0

2 FullyConnectedLayerSolved/kernel
,:*
2FullyConnectedLayerSolved/bias
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
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
s	variables
ttrainable_variables
uregularization_losses
w__call__
*x&call_and_return_all_conditional_losses
&x"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
5__inference_PredictionAreaRatio_layer_call_fn_1432904�
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
P__inference_PredictionAreaRatio_layer_call_and_return_conditional_losses_1432915�
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
,:*
2PredictionAreaRatio/kernel
&:$2PredictionAreaRatio/bias
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
{	variables
|trainable_variables
}regularization_losses
__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
2__inference_PredictionSolved_layer_call_fn_1432924�
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
M__inference_PredictionSolved_layer_call_and_return_conditional_losses_1432935�
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
):'
2PredictionSolved/kernel
#:!2PredictionSolved/bias
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
�
�trace_0
�trace_12�
(__inference_Output_layer_call_fn_1432940
(__inference_Output_layer_call_fn_1432945�
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
C__inference_Output_layer_call_and_return_conditional_losses_1432950
C__inference_Output_layer_call_and_return_conditional_losses_1432955�
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
V:T	2@transformer_encoder_15/Encoder-SelfAttentionLayer-1/query/kernel
P:N2>transformer_encoder_15/Encoder-SelfAttentionLayer-1/query/bias
T:R	2>transformer_encoder_15/Encoder-SelfAttentionLayer-1/key/kernel
N:L2<transformer_encoder_15/Encoder-SelfAttentionLayer-1/key/bias
V:T	2@transformer_encoder_15/Encoder-SelfAttentionLayer-1/value/kernel
P:N2>transformer_encoder_15/Encoder-SelfAttentionLayer-1/value/bias
a:_	2Ktransformer_encoder_15/Encoder-SelfAttentionLayer-1/attention_output/kernel
W:U	2Itransformer_encoder_15/Encoder-SelfAttentionLayer-1/attention_output/bias
K:I	2=transformer_encoder_15/Encoder-1st-NormalizationLayer-1/gamma
J:H	2<transformer_encoder_15/Encoder-1st-NormalizationLayer-1/beta
K:I	2=transformer_encoder_15/Encoder-2nd-NormalizationLayer-1/gamma
J:H	2<transformer_encoder_15/Encoder-2nd-NormalizationLayer-1/beta
L:J		2:transformer_encoder_15/Encoder-FeedForwardLayer_1_1/kernel
F:D	28transformer_encoder_15/Encoder-FeedForwardLayer_1_1/bias
L:J		2:transformer_encoder_15/Encoder-FeedForwardLayer_2_1/kernel
F:D	28transformer_encoder_15/Encoder-FeedForwardLayer_2_1/bias
V:T	2@transformer_encoder_16/Encoder-SelfAttentionLayer-2/query/kernel
P:N2>transformer_encoder_16/Encoder-SelfAttentionLayer-2/query/bias
T:R	2>transformer_encoder_16/Encoder-SelfAttentionLayer-2/key/kernel
N:L2<transformer_encoder_16/Encoder-SelfAttentionLayer-2/key/bias
V:T	2@transformer_encoder_16/Encoder-SelfAttentionLayer-2/value/kernel
P:N2>transformer_encoder_16/Encoder-SelfAttentionLayer-2/value/bias
a:_	2Ktransformer_encoder_16/Encoder-SelfAttentionLayer-2/attention_output/kernel
W:U	2Itransformer_encoder_16/Encoder-SelfAttentionLayer-2/attention_output/bias
K:I	2=transformer_encoder_16/Encoder-1st-NormalizationLayer-2/gamma
J:H	2<transformer_encoder_16/Encoder-1st-NormalizationLayer-2/beta
K:I	2=transformer_encoder_16/Encoder-2nd-NormalizationLayer-2/gamma
J:H	2<transformer_encoder_16/Encoder-2nd-NormalizationLayer-2/beta
L:J		2:transformer_encoder_16/Encoder-FeedForwardLayer_1_2/kernel
F:D	28transformer_encoder_16/Encoder-FeedForwardLayer_1_2/bias
L:J		2:transformer_encoder_16/Encoder-FeedForwardLayer_2_2/kernel
F:D	28transformer_encoder_16/Encoder-FeedForwardLayer_2_2/bias
V:T	2@transformer_encoder_17/Encoder-SelfAttentionLayer-3/query/kernel
P:N2>transformer_encoder_17/Encoder-SelfAttentionLayer-3/query/bias
T:R	2>transformer_encoder_17/Encoder-SelfAttentionLayer-3/key/kernel
N:L2<transformer_encoder_17/Encoder-SelfAttentionLayer-3/key/bias
V:T	2@transformer_encoder_17/Encoder-SelfAttentionLayer-3/value/kernel
P:N2>transformer_encoder_17/Encoder-SelfAttentionLayer-3/value/bias
a:_	2Ktransformer_encoder_17/Encoder-SelfAttentionLayer-3/attention_output/kernel
W:U	2Itransformer_encoder_17/Encoder-SelfAttentionLayer-3/attention_output/bias
K:I	2=transformer_encoder_17/Encoder-1st-NormalizationLayer-3/gamma
J:H	2<transformer_encoder_17/Encoder-1st-NormalizationLayer-3/beta
K:I	2=transformer_encoder_17/Encoder-2nd-NormalizationLayer-3/gamma
J:H	2<transformer_encoder_17/Encoder-2nd-NormalizationLayer-3/beta
L:J		2:transformer_encoder_17/Encoder-FeedForwardLayer_1_3/kernel
F:D	28transformer_encoder_17/Encoder-FeedForwardLayer_1_3/bias
L:J		2:transformer_encoder_17/Encoder-FeedForwardLayer_2_3/kernel
F:D	28transformer_encoder_17/Encoder-FeedForwardLayer_2_3/bias
 "
trackable_list_wrapper
�
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
12
13
14"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
)__inference_model_5_layer_call_fn_1430820StackLevelInputFeaturesTimeLimitInput"�
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
)__inference_model_5_layer_call_fn_1430944StackLevelInputFeaturesTimeLimitInput"�
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
D__inference_model_5_layer_call_and_return_conditional_losses_1430011StackLevelInputFeaturesTimeLimitInput"�
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
D__inference_model_5_layer_call_and_return_conditional_losses_1430696StackLevelInputFeaturesTimeLimitInput"�
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
%__inference_signature_wrapper_1431389StackLevelInputFeaturesTimeLimitInput"�
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
.__inference_MaskingLayer_layer_call_fn_1431394inputs"�
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
I__inference_MaskingLayer_layer_call_and_return_conditional_losses_1431405inputs"�
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
$0
%1
&2
'3
(4
)5
*6
+7"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
8__inference_transformer_encoder_15_layer_call_fn_1431444inputs"�
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
8__inference_transformer_encoder_15_layer_call_fn_1431483inputs"�
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
S__inference_transformer_encoder_15_layer_call_and_return_conditional_losses_1431671inputs"�
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
S__inference_transformer_encoder_15_layer_call_and_return_conditional_losses_1431845inputs"�
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
�kernel
	�bias"
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
�kernel
	�bias"
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
�kernel
	�bias"
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
20
31
42
53
64
75
86
97"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
8__inference_transformer_encoder_16_layer_call_fn_1431884inputs"�
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
8__inference_transformer_encoder_16_layer_call_fn_1431923inputs"�
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
S__inference_transformer_encoder_16_layer_call_and_return_conditional_losses_1432111inputs"�
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
S__inference_transformer_encoder_16_layer_call_and_return_conditional_losses_1432285inputs"�
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
@0
A1
B2
C3
D4
E5
F6
G7"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
8__inference_transformer_encoder_17_layer_call_fn_1432324inputs"�
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
8__inference_transformer_encoder_17_layer_call_fn_1432363inputs"�
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
S__inference_transformer_encoder_17_layer_call_and_return_conditional_losses_1432551inputs"�
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
S__inference_transformer_encoder_17_layer_call_and_return_conditional_losses_1432725inputs"�
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
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
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
0__inference_FinalLayerNorm_layer_call_fn_1432734inputs"�
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
K__inference_FinalLayerNorm_layer_call_and_return_conditional_losses_1432776inputs"�
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
�B�
B__inference_ReduceStackDimensionViaSummation_layer_call_fn_1432781inputs"�
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
B__inference_ReduceStackDimensionViaSummation_layer_call_fn_1432786inputs"�
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
]__inference_ReduceStackDimensionViaSummation_layer_call_and_return_conditional_losses_1432794inputs"�
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
]__inference_ReduceStackDimensionViaSummation_layer_call_and_return_conditional_losses_1432802inputs"�
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
6__inference_StandardizeTimeLimit_layer_call_fn_1432807inputs"�
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
6__inference_StandardizeTimeLimit_layer_call_fn_1432812inputs"�
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
Q__inference_StandardizeTimeLimit_layer_call_and_return_conditional_losses_1432820inputs"�
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
Q__inference_StandardizeTimeLimit_layer_call_and_return_conditional_losses_1432828inputs"�
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
2__inference_ConcatenateLayer_layer_call_fn_1432834inputs_0inputs_1"�
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
M__inference_ConcatenateLayer_layer_call_and_return_conditional_losses_1432841inputs_0inputs_1"�
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
>__inference_FullyConnectedLayerAreaRatio_layer_call_fn_1432850inputs"�
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
Y__inference_FullyConnectedLayerAreaRatio_layer_call_and_return_conditional_losses_1432868inputs"�
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
;__inference_FullyConnectedLayerSolved_layer_call_fn_1432877inputs"�
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
V__inference_FullyConnectedLayerSolved_layer_call_and_return_conditional_losses_1432895inputs"�
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
5__inference_PredictionAreaRatio_layer_call_fn_1432904inputs"�
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
P__inference_PredictionAreaRatio_layer_call_and_return_conditional_losses_1432915inputs"�
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
2__inference_PredictionSolved_layer_call_fn_1432924inputs"�
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
M__inference_PredictionSolved_layer_call_and_return_conditional_losses_1432935inputs"�
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
(__inference_Output_layer_call_fn_1432940inputs"�
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
(__inference_Output_layer_call_fn_1432945inputs"�
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
C__inference_Output_layer_call_and_return_conditional_losses_1432950inputs"�
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
C__inference_Output_layer_call_and_return_conditional_losses_1432955inputs"�
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
�	variables
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
�layer_metrics
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
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
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
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
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
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
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
M__inference_ConcatenateLayer_layer_call_and_return_conditional_losses_1432841�Z�W
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
2__inference_ConcatenateLayer_layer_call_fn_1432834Z�W
P�M
K�H
"�
inputs_0���������	
"�
inputs_1���������
� "!�
unknown���������
�
K__inference_FinalLayerNorm_layer_call_and_return_conditional_losses_1432776kOP3�0
)�&
$�!
inputs���������P	
� "0�-
&�#
tensor_0���������P	
� �
0__inference_FinalLayerNorm_layer_call_fn_1432734`OP3�0
)�&
$�!
inputs���������P	
� "%�"
unknown���������P	�
Y__inference_FullyConnectedLayerAreaRatio_layer_call_and_return_conditional_losses_1432868cij/�,
%�"
 �
inputs���������

� ",�)
"�
tensor_0���������

� �
>__inference_FullyConnectedLayerAreaRatio_layer_call_fn_1432850Xij/�,
%�"
 �
inputs���������

� "!�
unknown���������
�
V__inference_FullyConnectedLayerSolved_layer_call_and_return_conditional_losses_1432895cqr/�,
%�"
 �
inputs���������

� ",�)
"�
tensor_0���������

� �
;__inference_FullyConnectedLayerSolved_layer_call_fn_1432877Xqr/�,
%�"
 �
inputs���������

� "!�
unknown���������
�
I__inference_MaskingLayer_layer_call_and_return_conditional_losses_1431405g3�0
)�&
$�!
inputs���������P	
� "0�-
&�#
tensor_0���������P	
� �
.__inference_MaskingLayer_layer_call_fn_1431394\3�0
)�&
$�!
inputs���������P	
� "%�"
unknown���������P	�
C__inference_Output_layer_call_and_return_conditional_losses_1432950X7�4
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
C__inference_Output_layer_call_and_return_conditional_losses_1432955X7�4
-�*
 �
inputs���������

 
p 
� "�
�
tensor_0
� y
(__inference_Output_layer_call_fn_1432940M7�4
-�*
 �
inputs���������

 
p
� "�
unknowny
(__inference_Output_layer_call_fn_1432945M7�4
-�*
 �
inputs���������

 
p 
� "�
unknown�
P__inference_PredictionAreaRatio_layer_call_and_return_conditional_losses_1432915cyz/�,
%�"
 �
inputs���������

� ",�)
"�
tensor_0���������
� �
5__inference_PredictionAreaRatio_layer_call_fn_1432904Xyz/�,
%�"
 �
inputs���������

� "!�
unknown����������
M__inference_PredictionSolved_layer_call_and_return_conditional_losses_1432935e��/�,
%�"
 �
inputs���������

� ",�)
"�
tensor_0���������
� �
2__inference_PredictionSolved_layer_call_fn_1432924Z��/�,
%�"
 �
inputs���������

� "!�
unknown����������
]__inference_ReduceStackDimensionViaSummation_layer_call_and_return_conditional_losses_1432794k;�8
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
]__inference_ReduceStackDimensionViaSummation_layer_call_and_return_conditional_losses_1432802k;�8
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
B__inference_ReduceStackDimensionViaSummation_layer_call_fn_1432781`;�8
1�.
$�!
inputs���������P	

 
p
� "!�
unknown���������	�
B__inference_ReduceStackDimensionViaSummation_layer_call_fn_1432786`;�8
1�.
$�!
inputs���������P	

 
p 
� "!�
unknown���������	�
Q__inference_StandardizeTimeLimit_layer_call_and_return_conditional_losses_1432820g7�4
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
Q__inference_StandardizeTimeLimit_layer_call_and_return_conditional_losses_1432828g7�4
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
6__inference_StandardizeTimeLimit_layer_call_fn_1432807\7�4
-�*
 �
inputs���������

 
p
� "!�
unknown����������
6__inference_StandardizeTimeLimit_layer_call_fn_1432812\7�4
-�*
 �
inputs���������

 
p 
� "!�
unknown����������
"__inference__wrapped_model_1429166�l������������������������������������������������OPqrij��yzs�p
i�f
d�a
5�2
StackLevelInputFeatures���������P	
(�%
TimeLimitInput���������
� "A�>

Output_1�
output_1

Output�
output�
D__inference_model_5_layer_call_and_return_conditional_losses_1430011�l������������������������������������������������OPqrij��yz{�x
q�n
d�a
5�2
StackLevelInputFeatures���������P	
(�%
TimeLimitInput���������
p

 
� ";�8
1�.
�

tensor_0_0
�

tensor_0_1
� �
D__inference_model_5_layer_call_and_return_conditional_losses_1430696�l������������������������������������������������OPqrij��yz{�x
q�n
d�a
5�2
StackLevelInputFeatures���������P	
(�%
TimeLimitInput���������
p 

 
� ";�8
1�.
�

tensor_0_0
�

tensor_0_1
� �
)__inference_model_5_layer_call_fn_1430820�l������������������������������������������������OPqrij��yz{�x
q�n
d�a
5�2
StackLevelInputFeatures���������P	
(�%
TimeLimitInput���������
p

 
� "-�*
�
tensor_0
�
tensor_1�
)__inference_model_5_layer_call_fn_1430944�l������������������������������������������������OPqrij��yz{�x
q�n
d�a
5�2
StackLevelInputFeatures���������P	
(�%
TimeLimitInput���������
p 

 
� "-�*
�
tensor_0
�
tensor_1�
%__inference_signature_wrapper_1431389�l������������������������������������������������OPqrij��yz���
� 
���
P
StackLevelInputFeatures5�2
stacklevelinputfeatures���������P	
:
TimeLimitInput(�%
timelimitinput���������"A�>

Output_1�
output_1

Output�
output�
S__inference_transformer_encoder_15_layer_call_and_return_conditional_losses_1431671� ����������������C�@
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
S__inference_transformer_encoder_15_layer_call_and_return_conditional_losses_1431845� ����������������C�@
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
8__inference_transformer_encoder_15_layer_call_fn_1431444� ����������������C�@
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
8__inference_transformer_encoder_15_layer_call_fn_1431483� ����������������C�@
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
S__inference_transformer_encoder_16_layer_call_and_return_conditional_losses_1432111� ����������������C�@
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
S__inference_transformer_encoder_16_layer_call_and_return_conditional_losses_1432285� ����������������C�@
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
8__inference_transformer_encoder_16_layer_call_fn_1431884� ����������������C�@
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
8__inference_transformer_encoder_16_layer_call_fn_1431923� ����������������C�@
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
S__inference_transformer_encoder_17_layer_call_and_return_conditional_losses_1432551� ����������������C�@
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
S__inference_transformer_encoder_17_layer_call_and_return_conditional_losses_1432725� ����������������C�@
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
8__inference_transformer_encoder_17_layer_call_fn_1432324� ����������������C�@
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
8__inference_transformer_encoder_17_layer_call_fn_1432363� ����������������C�@
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