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
8transformer_encoder_20/Encoder-FeedForwardLayer_2_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*I
shared_name:8transformer_encoder_20/Encoder-FeedForwardLayer_2_3/bias
�
Ltransformer_encoder_20/Encoder-FeedForwardLayer_2_3/bias/Read/ReadVariableOpReadVariableOp8transformer_encoder_20/Encoder-FeedForwardLayer_2_3/bias*
_output_shapes
:	*
dtype0
�
:transformer_encoder_20/Encoder-FeedForwardLayer_2_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:		*K
shared_name<:transformer_encoder_20/Encoder-FeedForwardLayer_2_3/kernel
�
Ntransformer_encoder_20/Encoder-FeedForwardLayer_2_3/kernel/Read/ReadVariableOpReadVariableOp:transformer_encoder_20/Encoder-FeedForwardLayer_2_3/kernel*
_output_shapes

:		*
dtype0
�
8transformer_encoder_20/Encoder-FeedForwardLayer_1_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*I
shared_name:8transformer_encoder_20/Encoder-FeedForwardLayer_1_3/bias
�
Ltransformer_encoder_20/Encoder-FeedForwardLayer_1_3/bias/Read/ReadVariableOpReadVariableOp8transformer_encoder_20/Encoder-FeedForwardLayer_1_3/bias*
_output_shapes
:	*
dtype0
�
:transformer_encoder_20/Encoder-FeedForwardLayer_1_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:		*K
shared_name<:transformer_encoder_20/Encoder-FeedForwardLayer_1_3/kernel
�
Ntransformer_encoder_20/Encoder-FeedForwardLayer_1_3/kernel/Read/ReadVariableOpReadVariableOp:transformer_encoder_20/Encoder-FeedForwardLayer_1_3/kernel*
_output_shapes

:		*
dtype0
�
<transformer_encoder_20/Encoder-2nd-NormalizationLayer-3/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*M
shared_name><transformer_encoder_20/Encoder-2nd-NormalizationLayer-3/beta
�
Ptransformer_encoder_20/Encoder-2nd-NormalizationLayer-3/beta/Read/ReadVariableOpReadVariableOp<transformer_encoder_20/Encoder-2nd-NormalizationLayer-3/beta*
_output_shapes
:	*
dtype0
�
=transformer_encoder_20/Encoder-2nd-NormalizationLayer-3/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*N
shared_name?=transformer_encoder_20/Encoder-2nd-NormalizationLayer-3/gamma
�
Qtransformer_encoder_20/Encoder-2nd-NormalizationLayer-3/gamma/Read/ReadVariableOpReadVariableOp=transformer_encoder_20/Encoder-2nd-NormalizationLayer-3/gamma*
_output_shapes
:	*
dtype0
�
<transformer_encoder_20/Encoder-1st-NormalizationLayer-3/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*M
shared_name><transformer_encoder_20/Encoder-1st-NormalizationLayer-3/beta
�
Ptransformer_encoder_20/Encoder-1st-NormalizationLayer-3/beta/Read/ReadVariableOpReadVariableOp<transformer_encoder_20/Encoder-1st-NormalizationLayer-3/beta*
_output_shapes
:	*
dtype0
�
=transformer_encoder_20/Encoder-1st-NormalizationLayer-3/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*N
shared_name?=transformer_encoder_20/Encoder-1st-NormalizationLayer-3/gamma
�
Qtransformer_encoder_20/Encoder-1st-NormalizationLayer-3/gamma/Read/ReadVariableOpReadVariableOp=transformer_encoder_20/Encoder-1st-NormalizationLayer-3/gamma*
_output_shapes
:	*
dtype0
�
Itransformer_encoder_20/Encoder-SelfAttentionLayer-3/attention_output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*Z
shared_nameKItransformer_encoder_20/Encoder-SelfAttentionLayer-3/attention_output/bias
�
]transformer_encoder_20/Encoder-SelfAttentionLayer-3/attention_output/bias/Read/ReadVariableOpReadVariableOpItransformer_encoder_20/Encoder-SelfAttentionLayer-3/attention_output/bias*
_output_shapes
:	*
dtype0
�
Ktransformer_encoder_20/Encoder-SelfAttentionLayer-3/attention_output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*\
shared_nameMKtransformer_encoder_20/Encoder-SelfAttentionLayer-3/attention_output/kernel
�
_transformer_encoder_20/Encoder-SelfAttentionLayer-3/attention_output/kernel/Read/ReadVariableOpReadVariableOpKtransformer_encoder_20/Encoder-SelfAttentionLayer-3/attention_output/kernel*"
_output_shapes
:	*
dtype0
�
>transformer_encoder_20/Encoder-SelfAttentionLayer-3/value/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*O
shared_name@>transformer_encoder_20/Encoder-SelfAttentionLayer-3/value/bias
�
Rtransformer_encoder_20/Encoder-SelfAttentionLayer-3/value/bias/Read/ReadVariableOpReadVariableOp>transformer_encoder_20/Encoder-SelfAttentionLayer-3/value/bias*
_output_shapes

:*
dtype0
�
@transformer_encoder_20/Encoder-SelfAttentionLayer-3/value/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*Q
shared_nameB@transformer_encoder_20/Encoder-SelfAttentionLayer-3/value/kernel
�
Ttransformer_encoder_20/Encoder-SelfAttentionLayer-3/value/kernel/Read/ReadVariableOpReadVariableOp@transformer_encoder_20/Encoder-SelfAttentionLayer-3/value/kernel*"
_output_shapes
:	*
dtype0
�
<transformer_encoder_20/Encoder-SelfAttentionLayer-3/key/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*M
shared_name><transformer_encoder_20/Encoder-SelfAttentionLayer-3/key/bias
�
Ptransformer_encoder_20/Encoder-SelfAttentionLayer-3/key/bias/Read/ReadVariableOpReadVariableOp<transformer_encoder_20/Encoder-SelfAttentionLayer-3/key/bias*
_output_shapes

:*
dtype0
�
>transformer_encoder_20/Encoder-SelfAttentionLayer-3/key/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*O
shared_name@>transformer_encoder_20/Encoder-SelfAttentionLayer-3/key/kernel
�
Rtransformer_encoder_20/Encoder-SelfAttentionLayer-3/key/kernel/Read/ReadVariableOpReadVariableOp>transformer_encoder_20/Encoder-SelfAttentionLayer-3/key/kernel*"
_output_shapes
:	*
dtype0
�
>transformer_encoder_20/Encoder-SelfAttentionLayer-3/query/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*O
shared_name@>transformer_encoder_20/Encoder-SelfAttentionLayer-3/query/bias
�
Rtransformer_encoder_20/Encoder-SelfAttentionLayer-3/query/bias/Read/ReadVariableOpReadVariableOp>transformer_encoder_20/Encoder-SelfAttentionLayer-3/query/bias*
_output_shapes

:*
dtype0
�
@transformer_encoder_20/Encoder-SelfAttentionLayer-3/query/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*Q
shared_nameB@transformer_encoder_20/Encoder-SelfAttentionLayer-3/query/kernel
�
Ttransformer_encoder_20/Encoder-SelfAttentionLayer-3/query/kernel/Read/ReadVariableOpReadVariableOp@transformer_encoder_20/Encoder-SelfAttentionLayer-3/query/kernel*"
_output_shapes
:	*
dtype0
�
8transformer_encoder_19/Encoder-FeedForwardLayer_2_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*I
shared_name:8transformer_encoder_19/Encoder-FeedForwardLayer_2_2/bias
�
Ltransformer_encoder_19/Encoder-FeedForwardLayer_2_2/bias/Read/ReadVariableOpReadVariableOp8transformer_encoder_19/Encoder-FeedForwardLayer_2_2/bias*
_output_shapes
:	*
dtype0
�
:transformer_encoder_19/Encoder-FeedForwardLayer_2_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:		*K
shared_name<:transformer_encoder_19/Encoder-FeedForwardLayer_2_2/kernel
�
Ntransformer_encoder_19/Encoder-FeedForwardLayer_2_2/kernel/Read/ReadVariableOpReadVariableOp:transformer_encoder_19/Encoder-FeedForwardLayer_2_2/kernel*
_output_shapes

:		*
dtype0
�
8transformer_encoder_19/Encoder-FeedForwardLayer_1_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*I
shared_name:8transformer_encoder_19/Encoder-FeedForwardLayer_1_2/bias
�
Ltransformer_encoder_19/Encoder-FeedForwardLayer_1_2/bias/Read/ReadVariableOpReadVariableOp8transformer_encoder_19/Encoder-FeedForwardLayer_1_2/bias*
_output_shapes
:	*
dtype0
�
:transformer_encoder_19/Encoder-FeedForwardLayer_1_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:		*K
shared_name<:transformer_encoder_19/Encoder-FeedForwardLayer_1_2/kernel
�
Ntransformer_encoder_19/Encoder-FeedForwardLayer_1_2/kernel/Read/ReadVariableOpReadVariableOp:transformer_encoder_19/Encoder-FeedForwardLayer_1_2/kernel*
_output_shapes

:		*
dtype0
�
<transformer_encoder_19/Encoder-2nd-NormalizationLayer-2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*M
shared_name><transformer_encoder_19/Encoder-2nd-NormalizationLayer-2/beta
�
Ptransformer_encoder_19/Encoder-2nd-NormalizationLayer-2/beta/Read/ReadVariableOpReadVariableOp<transformer_encoder_19/Encoder-2nd-NormalizationLayer-2/beta*
_output_shapes
:	*
dtype0
�
=transformer_encoder_19/Encoder-2nd-NormalizationLayer-2/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*N
shared_name?=transformer_encoder_19/Encoder-2nd-NormalizationLayer-2/gamma
�
Qtransformer_encoder_19/Encoder-2nd-NormalizationLayer-2/gamma/Read/ReadVariableOpReadVariableOp=transformer_encoder_19/Encoder-2nd-NormalizationLayer-2/gamma*
_output_shapes
:	*
dtype0
�
<transformer_encoder_19/Encoder-1st-NormalizationLayer-2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*M
shared_name><transformer_encoder_19/Encoder-1st-NormalizationLayer-2/beta
�
Ptransformer_encoder_19/Encoder-1st-NormalizationLayer-2/beta/Read/ReadVariableOpReadVariableOp<transformer_encoder_19/Encoder-1st-NormalizationLayer-2/beta*
_output_shapes
:	*
dtype0
�
=transformer_encoder_19/Encoder-1st-NormalizationLayer-2/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*N
shared_name?=transformer_encoder_19/Encoder-1st-NormalizationLayer-2/gamma
�
Qtransformer_encoder_19/Encoder-1st-NormalizationLayer-2/gamma/Read/ReadVariableOpReadVariableOp=transformer_encoder_19/Encoder-1st-NormalizationLayer-2/gamma*
_output_shapes
:	*
dtype0
�
Itransformer_encoder_19/Encoder-SelfAttentionLayer-2/attention_output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*Z
shared_nameKItransformer_encoder_19/Encoder-SelfAttentionLayer-2/attention_output/bias
�
]transformer_encoder_19/Encoder-SelfAttentionLayer-2/attention_output/bias/Read/ReadVariableOpReadVariableOpItransformer_encoder_19/Encoder-SelfAttentionLayer-2/attention_output/bias*
_output_shapes
:	*
dtype0
�
Ktransformer_encoder_19/Encoder-SelfAttentionLayer-2/attention_output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*\
shared_nameMKtransformer_encoder_19/Encoder-SelfAttentionLayer-2/attention_output/kernel
�
_transformer_encoder_19/Encoder-SelfAttentionLayer-2/attention_output/kernel/Read/ReadVariableOpReadVariableOpKtransformer_encoder_19/Encoder-SelfAttentionLayer-2/attention_output/kernel*"
_output_shapes
:	*
dtype0
�
>transformer_encoder_19/Encoder-SelfAttentionLayer-2/value/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*O
shared_name@>transformer_encoder_19/Encoder-SelfAttentionLayer-2/value/bias
�
Rtransformer_encoder_19/Encoder-SelfAttentionLayer-2/value/bias/Read/ReadVariableOpReadVariableOp>transformer_encoder_19/Encoder-SelfAttentionLayer-2/value/bias*
_output_shapes

:*
dtype0
�
@transformer_encoder_19/Encoder-SelfAttentionLayer-2/value/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*Q
shared_nameB@transformer_encoder_19/Encoder-SelfAttentionLayer-2/value/kernel
�
Ttransformer_encoder_19/Encoder-SelfAttentionLayer-2/value/kernel/Read/ReadVariableOpReadVariableOp@transformer_encoder_19/Encoder-SelfAttentionLayer-2/value/kernel*"
_output_shapes
:	*
dtype0
�
<transformer_encoder_19/Encoder-SelfAttentionLayer-2/key/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*M
shared_name><transformer_encoder_19/Encoder-SelfAttentionLayer-2/key/bias
�
Ptransformer_encoder_19/Encoder-SelfAttentionLayer-2/key/bias/Read/ReadVariableOpReadVariableOp<transformer_encoder_19/Encoder-SelfAttentionLayer-2/key/bias*
_output_shapes

:*
dtype0
�
>transformer_encoder_19/Encoder-SelfAttentionLayer-2/key/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*O
shared_name@>transformer_encoder_19/Encoder-SelfAttentionLayer-2/key/kernel
�
Rtransformer_encoder_19/Encoder-SelfAttentionLayer-2/key/kernel/Read/ReadVariableOpReadVariableOp>transformer_encoder_19/Encoder-SelfAttentionLayer-2/key/kernel*"
_output_shapes
:	*
dtype0
�
>transformer_encoder_19/Encoder-SelfAttentionLayer-2/query/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*O
shared_name@>transformer_encoder_19/Encoder-SelfAttentionLayer-2/query/bias
�
Rtransformer_encoder_19/Encoder-SelfAttentionLayer-2/query/bias/Read/ReadVariableOpReadVariableOp>transformer_encoder_19/Encoder-SelfAttentionLayer-2/query/bias*
_output_shapes

:*
dtype0
�
@transformer_encoder_19/Encoder-SelfAttentionLayer-2/query/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*Q
shared_nameB@transformer_encoder_19/Encoder-SelfAttentionLayer-2/query/kernel
�
Ttransformer_encoder_19/Encoder-SelfAttentionLayer-2/query/kernel/Read/ReadVariableOpReadVariableOp@transformer_encoder_19/Encoder-SelfAttentionLayer-2/query/kernel*"
_output_shapes
:	*
dtype0
�
8transformer_encoder_18/Encoder-FeedForwardLayer_2_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*I
shared_name:8transformer_encoder_18/Encoder-FeedForwardLayer_2_1/bias
�
Ltransformer_encoder_18/Encoder-FeedForwardLayer_2_1/bias/Read/ReadVariableOpReadVariableOp8transformer_encoder_18/Encoder-FeedForwardLayer_2_1/bias*
_output_shapes
:	*
dtype0
�
:transformer_encoder_18/Encoder-FeedForwardLayer_2_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:		*K
shared_name<:transformer_encoder_18/Encoder-FeedForwardLayer_2_1/kernel
�
Ntransformer_encoder_18/Encoder-FeedForwardLayer_2_1/kernel/Read/ReadVariableOpReadVariableOp:transformer_encoder_18/Encoder-FeedForwardLayer_2_1/kernel*
_output_shapes

:		*
dtype0
�
8transformer_encoder_18/Encoder-FeedForwardLayer_1_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*I
shared_name:8transformer_encoder_18/Encoder-FeedForwardLayer_1_1/bias
�
Ltransformer_encoder_18/Encoder-FeedForwardLayer_1_1/bias/Read/ReadVariableOpReadVariableOp8transformer_encoder_18/Encoder-FeedForwardLayer_1_1/bias*
_output_shapes
:	*
dtype0
�
:transformer_encoder_18/Encoder-FeedForwardLayer_1_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:		*K
shared_name<:transformer_encoder_18/Encoder-FeedForwardLayer_1_1/kernel
�
Ntransformer_encoder_18/Encoder-FeedForwardLayer_1_1/kernel/Read/ReadVariableOpReadVariableOp:transformer_encoder_18/Encoder-FeedForwardLayer_1_1/kernel*
_output_shapes

:		*
dtype0
�
<transformer_encoder_18/Encoder-2nd-NormalizationLayer-1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*M
shared_name><transformer_encoder_18/Encoder-2nd-NormalizationLayer-1/beta
�
Ptransformer_encoder_18/Encoder-2nd-NormalizationLayer-1/beta/Read/ReadVariableOpReadVariableOp<transformer_encoder_18/Encoder-2nd-NormalizationLayer-1/beta*
_output_shapes
:	*
dtype0
�
=transformer_encoder_18/Encoder-2nd-NormalizationLayer-1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*N
shared_name?=transformer_encoder_18/Encoder-2nd-NormalizationLayer-1/gamma
�
Qtransformer_encoder_18/Encoder-2nd-NormalizationLayer-1/gamma/Read/ReadVariableOpReadVariableOp=transformer_encoder_18/Encoder-2nd-NormalizationLayer-1/gamma*
_output_shapes
:	*
dtype0
�
<transformer_encoder_18/Encoder-1st-NormalizationLayer-1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*M
shared_name><transformer_encoder_18/Encoder-1st-NormalizationLayer-1/beta
�
Ptransformer_encoder_18/Encoder-1st-NormalizationLayer-1/beta/Read/ReadVariableOpReadVariableOp<transformer_encoder_18/Encoder-1st-NormalizationLayer-1/beta*
_output_shapes
:	*
dtype0
�
=transformer_encoder_18/Encoder-1st-NormalizationLayer-1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*N
shared_name?=transformer_encoder_18/Encoder-1st-NormalizationLayer-1/gamma
�
Qtransformer_encoder_18/Encoder-1st-NormalizationLayer-1/gamma/Read/ReadVariableOpReadVariableOp=transformer_encoder_18/Encoder-1st-NormalizationLayer-1/gamma*
_output_shapes
:	*
dtype0
�
Itransformer_encoder_18/Encoder-SelfAttentionLayer-1/attention_output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*Z
shared_nameKItransformer_encoder_18/Encoder-SelfAttentionLayer-1/attention_output/bias
�
]transformer_encoder_18/Encoder-SelfAttentionLayer-1/attention_output/bias/Read/ReadVariableOpReadVariableOpItransformer_encoder_18/Encoder-SelfAttentionLayer-1/attention_output/bias*
_output_shapes
:	*
dtype0
�
Ktransformer_encoder_18/Encoder-SelfAttentionLayer-1/attention_output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*\
shared_nameMKtransformer_encoder_18/Encoder-SelfAttentionLayer-1/attention_output/kernel
�
_transformer_encoder_18/Encoder-SelfAttentionLayer-1/attention_output/kernel/Read/ReadVariableOpReadVariableOpKtransformer_encoder_18/Encoder-SelfAttentionLayer-1/attention_output/kernel*"
_output_shapes
:	*
dtype0
�
>transformer_encoder_18/Encoder-SelfAttentionLayer-1/value/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*O
shared_name@>transformer_encoder_18/Encoder-SelfAttentionLayer-1/value/bias
�
Rtransformer_encoder_18/Encoder-SelfAttentionLayer-1/value/bias/Read/ReadVariableOpReadVariableOp>transformer_encoder_18/Encoder-SelfAttentionLayer-1/value/bias*
_output_shapes

:*
dtype0
�
@transformer_encoder_18/Encoder-SelfAttentionLayer-1/value/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*Q
shared_nameB@transformer_encoder_18/Encoder-SelfAttentionLayer-1/value/kernel
�
Ttransformer_encoder_18/Encoder-SelfAttentionLayer-1/value/kernel/Read/ReadVariableOpReadVariableOp@transformer_encoder_18/Encoder-SelfAttentionLayer-1/value/kernel*"
_output_shapes
:	*
dtype0
�
<transformer_encoder_18/Encoder-SelfAttentionLayer-1/key/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*M
shared_name><transformer_encoder_18/Encoder-SelfAttentionLayer-1/key/bias
�
Ptransformer_encoder_18/Encoder-SelfAttentionLayer-1/key/bias/Read/ReadVariableOpReadVariableOp<transformer_encoder_18/Encoder-SelfAttentionLayer-1/key/bias*
_output_shapes

:*
dtype0
�
>transformer_encoder_18/Encoder-SelfAttentionLayer-1/key/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*O
shared_name@>transformer_encoder_18/Encoder-SelfAttentionLayer-1/key/kernel
�
Rtransformer_encoder_18/Encoder-SelfAttentionLayer-1/key/kernel/Read/ReadVariableOpReadVariableOp>transformer_encoder_18/Encoder-SelfAttentionLayer-1/key/kernel*"
_output_shapes
:	*
dtype0
�
>transformer_encoder_18/Encoder-SelfAttentionLayer-1/query/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*O
shared_name@>transformer_encoder_18/Encoder-SelfAttentionLayer-1/query/bias
�
Rtransformer_encoder_18/Encoder-SelfAttentionLayer-1/query/bias/Read/ReadVariableOpReadVariableOp>transformer_encoder_18/Encoder-SelfAttentionLayer-1/query/bias*
_output_shapes

:*
dtype0
�
@transformer_encoder_18/Encoder-SelfAttentionLayer-1/query/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*Q
shared_nameB@transformer_encoder_18/Encoder-SelfAttentionLayer-1/query/kernel
�
Ttransformer_encoder_18/Encoder-SelfAttentionLayer-1/query/kernel/Read/ReadVariableOpReadVariableOp@transformer_encoder_18/Encoder-SelfAttentionLayer-1/query/kernel*"
_output_shapes
:	*
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
StatefulPartitionedCallStatefulPartitionedCall'serving_default_StackLevelInputFeaturesserving_default_TimeLimitInput=transformer_encoder_18/Encoder-1st-NormalizationLayer-1/gamma<transformer_encoder_18/Encoder-1st-NormalizationLayer-1/beta@transformer_encoder_18/Encoder-SelfAttentionLayer-1/query/kernel>transformer_encoder_18/Encoder-SelfAttentionLayer-1/query/bias>transformer_encoder_18/Encoder-SelfAttentionLayer-1/key/kernel<transformer_encoder_18/Encoder-SelfAttentionLayer-1/key/bias@transformer_encoder_18/Encoder-SelfAttentionLayer-1/value/kernel>transformer_encoder_18/Encoder-SelfAttentionLayer-1/value/biasKtransformer_encoder_18/Encoder-SelfAttentionLayer-1/attention_output/kernelItransformer_encoder_18/Encoder-SelfAttentionLayer-1/attention_output/bias=transformer_encoder_18/Encoder-2nd-NormalizationLayer-1/gamma<transformer_encoder_18/Encoder-2nd-NormalizationLayer-1/beta:transformer_encoder_18/Encoder-FeedForwardLayer_1_1/kernel8transformer_encoder_18/Encoder-FeedForwardLayer_1_1/bias:transformer_encoder_18/Encoder-FeedForwardLayer_2_1/kernel8transformer_encoder_18/Encoder-FeedForwardLayer_2_1/bias=transformer_encoder_19/Encoder-1st-NormalizationLayer-2/gamma<transformer_encoder_19/Encoder-1st-NormalizationLayer-2/beta@transformer_encoder_19/Encoder-SelfAttentionLayer-2/query/kernel>transformer_encoder_19/Encoder-SelfAttentionLayer-2/query/bias>transformer_encoder_19/Encoder-SelfAttentionLayer-2/key/kernel<transformer_encoder_19/Encoder-SelfAttentionLayer-2/key/bias@transformer_encoder_19/Encoder-SelfAttentionLayer-2/value/kernel>transformer_encoder_19/Encoder-SelfAttentionLayer-2/value/biasKtransformer_encoder_19/Encoder-SelfAttentionLayer-2/attention_output/kernelItransformer_encoder_19/Encoder-SelfAttentionLayer-2/attention_output/bias=transformer_encoder_19/Encoder-2nd-NormalizationLayer-2/gamma<transformer_encoder_19/Encoder-2nd-NormalizationLayer-2/beta:transformer_encoder_19/Encoder-FeedForwardLayer_1_2/kernel8transformer_encoder_19/Encoder-FeedForwardLayer_1_2/bias:transformer_encoder_19/Encoder-FeedForwardLayer_2_2/kernel8transformer_encoder_19/Encoder-FeedForwardLayer_2_2/bias=transformer_encoder_20/Encoder-1st-NormalizationLayer-3/gamma<transformer_encoder_20/Encoder-1st-NormalizationLayer-3/beta@transformer_encoder_20/Encoder-SelfAttentionLayer-3/query/kernel>transformer_encoder_20/Encoder-SelfAttentionLayer-3/query/bias>transformer_encoder_20/Encoder-SelfAttentionLayer-3/key/kernel<transformer_encoder_20/Encoder-SelfAttentionLayer-3/key/bias@transformer_encoder_20/Encoder-SelfAttentionLayer-3/value/kernel>transformer_encoder_20/Encoder-SelfAttentionLayer-3/value/biasKtransformer_encoder_20/Encoder-SelfAttentionLayer-3/attention_output/kernelItransformer_encoder_20/Encoder-SelfAttentionLayer-3/attention_output/bias=transformer_encoder_20/Encoder-2nd-NormalizationLayer-3/gamma<transformer_encoder_20/Encoder-2nd-NormalizationLayer-3/beta:transformer_encoder_20/Encoder-FeedForwardLayer_1_3/kernel8transformer_encoder_20/Encoder-FeedForwardLayer_1_3/bias:transformer_encoder_20/Encoder-FeedForwardLayer_2_3/kernel8transformer_encoder_20/Encoder-FeedForwardLayer_2_3/biasFinalLayerNorm/gammaFinalLayerNorm/beta#FullyConnectedLayerAreaRatio/kernel!FullyConnectedLayerAreaRatio/biasPredictionAreaRatio/kernelPredictionAreaRatio/bias*C
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
GPU2*0J 8� *.
f)R'
%__inference_signature_wrapper_1474303

NoOpNoOp
��
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�
value޵Bڵ Bҵ
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
sm
VARIABLE_VALUE#FullyConnectedLayerAreaRatio/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
oi
VARIABLE_VALUE!FullyConnectedLayerAreaRatio/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*
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
jd
VARIABLE_VALUEPredictionAreaRatio/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUEPredictionAreaRatio/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE*
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
�z
VARIABLE_VALUE@transformer_encoder_18/Encoder-SelfAttentionLayer-1/query/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE>transformer_encoder_18/Encoder-SelfAttentionLayer-1/query/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE>transformer_encoder_18/Encoder-SelfAttentionLayer-1/key/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUE<transformer_encoder_18/Encoder-SelfAttentionLayer-1/key/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
�z
VARIABLE_VALUE@transformer_encoder_18/Encoder-SelfAttentionLayer-1/value/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE>transformer_encoder_18/Encoder-SelfAttentionLayer-1/value/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEKtransformer_encoder_18/Encoder-SelfAttentionLayer-1/attention_output/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEItransformer_encoder_18/Encoder-SelfAttentionLayer-1/attention_output/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUE=transformer_encoder_18/Encoder-1st-NormalizationLayer-1/gamma&variables/8/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUE<transformer_encoder_18/Encoder-1st-NormalizationLayer-1/beta&variables/9/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE=transformer_encoder_18/Encoder-2nd-NormalizationLayer-1/gamma'variables/10/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUE<transformer_encoder_18/Encoder-2nd-NormalizationLayer-1/beta'variables/11/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUE:transformer_encoder_18/Encoder-FeedForwardLayer_1_1/kernel'variables/12/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUE8transformer_encoder_18/Encoder-FeedForwardLayer_1_1/bias'variables/13/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUE:transformer_encoder_18/Encoder-FeedForwardLayer_2_1/kernel'variables/14/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUE8transformer_encoder_18/Encoder-FeedForwardLayer_2_1/bias'variables/15/.ATTRIBUTES/VARIABLE_VALUE*
�{
VARIABLE_VALUE@transformer_encoder_19/Encoder-SelfAttentionLayer-2/query/kernel'variables/16/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE>transformer_encoder_19/Encoder-SelfAttentionLayer-2/query/bias'variables/17/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE>transformer_encoder_19/Encoder-SelfAttentionLayer-2/key/kernel'variables/18/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUE<transformer_encoder_19/Encoder-SelfAttentionLayer-2/key/bias'variables/19/.ATTRIBUTES/VARIABLE_VALUE*
�{
VARIABLE_VALUE@transformer_encoder_19/Encoder-SelfAttentionLayer-2/value/kernel'variables/20/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE>transformer_encoder_19/Encoder-SelfAttentionLayer-2/value/bias'variables/21/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEKtransformer_encoder_19/Encoder-SelfAttentionLayer-2/attention_output/kernel'variables/22/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEItransformer_encoder_19/Encoder-SelfAttentionLayer-2/attention_output/bias'variables/23/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE=transformer_encoder_19/Encoder-1st-NormalizationLayer-2/gamma'variables/24/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUE<transformer_encoder_19/Encoder-1st-NormalizationLayer-2/beta'variables/25/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE=transformer_encoder_19/Encoder-2nd-NormalizationLayer-2/gamma'variables/26/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUE<transformer_encoder_19/Encoder-2nd-NormalizationLayer-2/beta'variables/27/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUE:transformer_encoder_19/Encoder-FeedForwardLayer_1_2/kernel'variables/28/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUE8transformer_encoder_19/Encoder-FeedForwardLayer_1_2/bias'variables/29/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUE:transformer_encoder_19/Encoder-FeedForwardLayer_2_2/kernel'variables/30/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUE8transformer_encoder_19/Encoder-FeedForwardLayer_2_2/bias'variables/31/.ATTRIBUTES/VARIABLE_VALUE*
�{
VARIABLE_VALUE@transformer_encoder_20/Encoder-SelfAttentionLayer-3/query/kernel'variables/32/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE>transformer_encoder_20/Encoder-SelfAttentionLayer-3/query/bias'variables/33/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE>transformer_encoder_20/Encoder-SelfAttentionLayer-3/key/kernel'variables/34/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUE<transformer_encoder_20/Encoder-SelfAttentionLayer-3/key/bias'variables/35/.ATTRIBUTES/VARIABLE_VALUE*
�{
VARIABLE_VALUE@transformer_encoder_20/Encoder-SelfAttentionLayer-3/value/kernel'variables/36/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE>transformer_encoder_20/Encoder-SelfAttentionLayer-3/value/bias'variables/37/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEKtransformer_encoder_20/Encoder-SelfAttentionLayer-3/attention_output/kernel'variables/38/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEItransformer_encoder_20/Encoder-SelfAttentionLayer-3/attention_output/bias'variables/39/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE=transformer_encoder_20/Encoder-1st-NormalizationLayer-3/gamma'variables/40/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUE<transformer_encoder_20/Encoder-1st-NormalizationLayer-3/beta'variables/41/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE=transformer_encoder_20/Encoder-2nd-NormalizationLayer-3/gamma'variables/42/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUE<transformer_encoder_20/Encoder-2nd-NormalizationLayer-3/beta'variables/43/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUE:transformer_encoder_20/Encoder-FeedForwardLayer_1_3/kernel'variables/44/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUE8transformer_encoder_20/Encoder-FeedForwardLayer_1_3/bias'variables/45/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUE:transformer_encoder_20/Encoder-FeedForwardLayer_2_3/kernel'variables/46/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUE8transformer_encoder_20/Encoder-FeedForwardLayer_2_3/bias'variables/47/.ATTRIBUTES/VARIABLE_VALUE*
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
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameFinalLayerNorm/gammaFinalLayerNorm/beta#FullyConnectedLayerAreaRatio/kernel!FullyConnectedLayerAreaRatio/biasPredictionAreaRatio/kernelPredictionAreaRatio/bias@transformer_encoder_18/Encoder-SelfAttentionLayer-1/query/kernel>transformer_encoder_18/Encoder-SelfAttentionLayer-1/query/bias>transformer_encoder_18/Encoder-SelfAttentionLayer-1/key/kernel<transformer_encoder_18/Encoder-SelfAttentionLayer-1/key/bias@transformer_encoder_18/Encoder-SelfAttentionLayer-1/value/kernel>transformer_encoder_18/Encoder-SelfAttentionLayer-1/value/biasKtransformer_encoder_18/Encoder-SelfAttentionLayer-1/attention_output/kernelItransformer_encoder_18/Encoder-SelfAttentionLayer-1/attention_output/bias=transformer_encoder_18/Encoder-1st-NormalizationLayer-1/gamma<transformer_encoder_18/Encoder-1st-NormalizationLayer-1/beta=transformer_encoder_18/Encoder-2nd-NormalizationLayer-1/gamma<transformer_encoder_18/Encoder-2nd-NormalizationLayer-1/beta:transformer_encoder_18/Encoder-FeedForwardLayer_1_1/kernel8transformer_encoder_18/Encoder-FeedForwardLayer_1_1/bias:transformer_encoder_18/Encoder-FeedForwardLayer_2_1/kernel8transformer_encoder_18/Encoder-FeedForwardLayer_2_1/bias@transformer_encoder_19/Encoder-SelfAttentionLayer-2/query/kernel>transformer_encoder_19/Encoder-SelfAttentionLayer-2/query/bias>transformer_encoder_19/Encoder-SelfAttentionLayer-2/key/kernel<transformer_encoder_19/Encoder-SelfAttentionLayer-2/key/bias@transformer_encoder_19/Encoder-SelfAttentionLayer-2/value/kernel>transformer_encoder_19/Encoder-SelfAttentionLayer-2/value/biasKtransformer_encoder_19/Encoder-SelfAttentionLayer-2/attention_output/kernelItransformer_encoder_19/Encoder-SelfAttentionLayer-2/attention_output/bias=transformer_encoder_19/Encoder-1st-NormalizationLayer-2/gamma<transformer_encoder_19/Encoder-1st-NormalizationLayer-2/beta=transformer_encoder_19/Encoder-2nd-NormalizationLayer-2/gamma<transformer_encoder_19/Encoder-2nd-NormalizationLayer-2/beta:transformer_encoder_19/Encoder-FeedForwardLayer_1_2/kernel8transformer_encoder_19/Encoder-FeedForwardLayer_1_2/bias:transformer_encoder_19/Encoder-FeedForwardLayer_2_2/kernel8transformer_encoder_19/Encoder-FeedForwardLayer_2_2/bias@transformer_encoder_20/Encoder-SelfAttentionLayer-3/query/kernel>transformer_encoder_20/Encoder-SelfAttentionLayer-3/query/bias>transformer_encoder_20/Encoder-SelfAttentionLayer-3/key/kernel<transformer_encoder_20/Encoder-SelfAttentionLayer-3/key/bias@transformer_encoder_20/Encoder-SelfAttentionLayer-3/value/kernel>transformer_encoder_20/Encoder-SelfAttentionLayer-3/value/biasKtransformer_encoder_20/Encoder-SelfAttentionLayer-3/attention_output/kernelItransformer_encoder_20/Encoder-SelfAttentionLayer-3/attention_output/bias=transformer_encoder_20/Encoder-1st-NormalizationLayer-3/gamma<transformer_encoder_20/Encoder-1st-NormalizationLayer-3/beta=transformer_encoder_20/Encoder-2nd-NormalizationLayer-3/gamma<transformer_encoder_20/Encoder-2nd-NormalizationLayer-3/beta:transformer_encoder_20/Encoder-FeedForwardLayer_1_3/kernel8transformer_encoder_20/Encoder-FeedForwardLayer_1_3/bias:transformer_encoder_20/Encoder-FeedForwardLayer_2_3/kernel8transformer_encoder_20/Encoder-FeedForwardLayer_2_3/biasConst*C
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
GPU2*0J 8� *)
f$R"
 __inference__traced_save_1476169
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameFinalLayerNorm/gammaFinalLayerNorm/beta#FullyConnectedLayerAreaRatio/kernel!FullyConnectedLayerAreaRatio/biasPredictionAreaRatio/kernelPredictionAreaRatio/bias@transformer_encoder_18/Encoder-SelfAttentionLayer-1/query/kernel>transformer_encoder_18/Encoder-SelfAttentionLayer-1/query/bias>transformer_encoder_18/Encoder-SelfAttentionLayer-1/key/kernel<transformer_encoder_18/Encoder-SelfAttentionLayer-1/key/bias@transformer_encoder_18/Encoder-SelfAttentionLayer-1/value/kernel>transformer_encoder_18/Encoder-SelfAttentionLayer-1/value/biasKtransformer_encoder_18/Encoder-SelfAttentionLayer-1/attention_output/kernelItransformer_encoder_18/Encoder-SelfAttentionLayer-1/attention_output/bias=transformer_encoder_18/Encoder-1st-NormalizationLayer-1/gamma<transformer_encoder_18/Encoder-1st-NormalizationLayer-1/beta=transformer_encoder_18/Encoder-2nd-NormalizationLayer-1/gamma<transformer_encoder_18/Encoder-2nd-NormalizationLayer-1/beta:transformer_encoder_18/Encoder-FeedForwardLayer_1_1/kernel8transformer_encoder_18/Encoder-FeedForwardLayer_1_1/bias:transformer_encoder_18/Encoder-FeedForwardLayer_2_1/kernel8transformer_encoder_18/Encoder-FeedForwardLayer_2_1/bias@transformer_encoder_19/Encoder-SelfAttentionLayer-2/query/kernel>transformer_encoder_19/Encoder-SelfAttentionLayer-2/query/bias>transformer_encoder_19/Encoder-SelfAttentionLayer-2/key/kernel<transformer_encoder_19/Encoder-SelfAttentionLayer-2/key/bias@transformer_encoder_19/Encoder-SelfAttentionLayer-2/value/kernel>transformer_encoder_19/Encoder-SelfAttentionLayer-2/value/biasKtransformer_encoder_19/Encoder-SelfAttentionLayer-2/attention_output/kernelItransformer_encoder_19/Encoder-SelfAttentionLayer-2/attention_output/bias=transformer_encoder_19/Encoder-1st-NormalizationLayer-2/gamma<transformer_encoder_19/Encoder-1st-NormalizationLayer-2/beta=transformer_encoder_19/Encoder-2nd-NormalizationLayer-2/gamma<transformer_encoder_19/Encoder-2nd-NormalizationLayer-2/beta:transformer_encoder_19/Encoder-FeedForwardLayer_1_2/kernel8transformer_encoder_19/Encoder-FeedForwardLayer_1_2/bias:transformer_encoder_19/Encoder-FeedForwardLayer_2_2/kernel8transformer_encoder_19/Encoder-FeedForwardLayer_2_2/bias@transformer_encoder_20/Encoder-SelfAttentionLayer-3/query/kernel>transformer_encoder_20/Encoder-SelfAttentionLayer-3/query/bias>transformer_encoder_20/Encoder-SelfAttentionLayer-3/key/kernel<transformer_encoder_20/Encoder-SelfAttentionLayer-3/key/bias@transformer_encoder_20/Encoder-SelfAttentionLayer-3/value/kernel>transformer_encoder_20/Encoder-SelfAttentionLayer-3/value/biasKtransformer_encoder_20/Encoder-SelfAttentionLayer-3/attention_output/kernelItransformer_encoder_20/Encoder-SelfAttentionLayer-3/attention_output/bias=transformer_encoder_20/Encoder-1st-NormalizationLayer-3/gamma<transformer_encoder_20/Encoder-1st-NormalizationLayer-3/beta=transformer_encoder_20/Encoder-2nd-NormalizationLayer-3/gamma<transformer_encoder_20/Encoder-2nd-NormalizationLayer-3/beta:transformer_encoder_20/Encoder-FeedForwardLayer_1_3/kernel8transformer_encoder_20/Encoder-FeedForwardLayer_1_3/bias:transformer_encoder_20/Encoder-FeedForwardLayer_2_3/kernel8transformer_encoder_20/Encoder-FeedForwardLayer_2_3/bias*B
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
GPU2*0J 8� *,
f'R%
#__inference__traced_restore_1476340��.
�
^
B__inference_ReduceStackDimensionViaSummation_layer_call_fn_1475700

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
]__inference_ReduceStackDimensionViaSummation_layer_call_and_return_conditional_losses_1473628`
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
�g
�
D__inference_model_6_layer_call_and_return_conditional_losses_1472986
stacklevelinputfeatures
timelimitinput,
transformer_encoder_18_1472388:	,
transformer_encoder_18_1472390:	4
transformer_encoder_18_1472392:	0
transformer_encoder_18_1472394:4
transformer_encoder_18_1472396:	0
transformer_encoder_18_1472398:4
transformer_encoder_18_1472400:	0
transformer_encoder_18_1472402:4
transformer_encoder_18_1472404:	,
transformer_encoder_18_1472406:	,
transformer_encoder_18_1472408:	,
transformer_encoder_18_1472410:	0
transformer_encoder_18_1472412:		,
transformer_encoder_18_1472414:	0
transformer_encoder_18_1472416:		,
transformer_encoder_18_1472418:	,
transformer_encoder_19_1472610:	,
transformer_encoder_19_1472612:	4
transformer_encoder_19_1472614:	0
transformer_encoder_19_1472616:4
transformer_encoder_19_1472618:	0
transformer_encoder_19_1472620:4
transformer_encoder_19_1472622:	0
transformer_encoder_19_1472624:4
transformer_encoder_19_1472626:	,
transformer_encoder_19_1472628:	,
transformer_encoder_19_1472630:	,
transformer_encoder_19_1472632:	0
transformer_encoder_19_1472634:		,
transformer_encoder_19_1472636:	0
transformer_encoder_19_1472638:		,
transformer_encoder_19_1472640:	,
transformer_encoder_20_1472832:	,
transformer_encoder_20_1472834:	4
transformer_encoder_20_1472836:	0
transformer_encoder_20_1472838:4
transformer_encoder_20_1472840:	0
transformer_encoder_20_1472842:4
transformer_encoder_20_1472844:	0
transformer_encoder_20_1472846:4
transformer_encoder_20_1472848:	,
transformer_encoder_20_1472850:	,
transformer_encoder_20_1472852:	,
transformer_encoder_20_1472854:	0
transformer_encoder_20_1472856:		,
transformer_encoder_20_1472858:	0
transformer_encoder_20_1472860:		,
transformer_encoder_20_1472862:	$
finallayernorm_1472908:	$
finallayernorm_1472910:	6
$fullyconnectedlayerarearatio_1472958:

2
$fullyconnectedlayerarearatio_1472960:
-
predictionarearatio_1472974:
)
predictionarearatio_1472976:
identity��&FinalLayerNorm/StatefulPartitionedCall�4FullyConnectedLayerAreaRatio/StatefulPartitionedCall�+PredictionAreaRatio/StatefulPartitionedCall�.transformer_encoder_18/StatefulPartitionedCall�.transformer_encoder_19/StatefulPartitionedCall�.transformer_encoder_20/StatefulPartitionedCall�
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
I__inference_MaskingLayer_layer_call_and_return_conditional_losses_1472197�
transformer_encoder_18/CastCast%MaskingLayer/PartitionedCall:output:0*

DstT0*

SrcT0*+
_output_shapes
:���������P	�
.transformer_encoder_18/StatefulPartitionedCallStatefulPartitionedCalltransformer_encoder_18/Cast:y:0transformer_encoder_18_1472388transformer_encoder_18_1472390transformer_encoder_18_1472392transformer_encoder_18_1472394transformer_encoder_18_1472396transformer_encoder_18_1472398transformer_encoder_18_1472400transformer_encoder_18_1472402transformer_encoder_18_1472404transformer_encoder_18_1472406transformer_encoder_18_1472408transformer_encoder_18_1472410transformer_encoder_18_1472412transformer_encoder_18_1472414transformer_encoder_18_1472416transformer_encoder_18_1472418*
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
S__inference_transformer_encoder_18_layer_call_and_return_conditional_losses_1472387�
.transformer_encoder_19/StatefulPartitionedCallStatefulPartitionedCall7transformer_encoder_18/StatefulPartitionedCall:output:0transformer_encoder_19_1472610transformer_encoder_19_1472612transformer_encoder_19_1472614transformer_encoder_19_1472616transformer_encoder_19_1472618transformer_encoder_19_1472620transformer_encoder_19_1472622transformer_encoder_19_1472624transformer_encoder_19_1472626transformer_encoder_19_1472628transformer_encoder_19_1472630transformer_encoder_19_1472632transformer_encoder_19_1472634transformer_encoder_19_1472636transformer_encoder_19_1472638transformer_encoder_19_1472640*
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
S__inference_transformer_encoder_19_layer_call_and_return_conditional_losses_1472609�
.transformer_encoder_20/StatefulPartitionedCallStatefulPartitionedCall7transformer_encoder_19/StatefulPartitionedCall:output:0transformer_encoder_20_1472832transformer_encoder_20_1472834transformer_encoder_20_1472836transformer_encoder_20_1472838transformer_encoder_20_1472840transformer_encoder_20_1472842transformer_encoder_20_1472844transformer_encoder_20_1472846transformer_encoder_20_1472848transformer_encoder_20_1472850transformer_encoder_20_1472852transformer_encoder_20_1472854transformer_encoder_20_1472856transformer_encoder_20_1472858transformer_encoder_20_1472860transformer_encoder_20_1472862*
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
S__inference_transformer_encoder_20_layer_call_and_return_conditional_losses_1472831�
&FinalLayerNorm/StatefulPartitionedCallStatefulPartitionedCall7transformer_encoder_20/StatefulPartitionedCall:output:0finallayernorm_1472908finallayernorm_1472910*
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
K__inference_FinalLayerNorm_layer_call_and_return_conditional_losses_1472907�
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
]__inference_ReduceStackDimensionViaSummation_layer_call_and_return_conditional_losses_1472920r
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
Q__inference_StandardizeTimeLimit_layer_call_and_return_conditional_losses_1472930�
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
M__inference_ConcatenateLayer_layer_call_and_return_conditional_losses_1472938�
4FullyConnectedLayerAreaRatio/StatefulPartitionedCallStatefulPartitionedCall)ConcatenateLayer/PartitionedCall:output:0$fullyconnectedlayerarearatio_1472958$fullyconnectedlayerarearatio_1472960*
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
Y__inference_FullyConnectedLayerAreaRatio_layer_call_and_return_conditional_losses_1472957�
+PredictionAreaRatio/StatefulPartitionedCallStatefulPartitionedCall=FullyConnectedLayerAreaRatio/StatefulPartitionedCall:output:0predictionarearatio_1472974predictionarearatio_1472976*
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
P__inference_PredictionAreaRatio_layer_call_and_return_conditional_losses_1472973�
Output/PartitionedCallPartitionedCall4PredictionAreaRatio/StatefulPartitionedCall:output:0*
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
C__inference_Output_layer_call_and_return_conditional_losses_1472983_
IdentityIdentityOutput/PartitionedCall:output:0^NoOp*
T0*
_output_shapes
:�
NoOpNoOp'^FinalLayerNorm/StatefulPartitionedCall5^FullyConnectedLayerAreaRatio/StatefulPartitionedCall,^PredictionAreaRatio/StatefulPartitionedCall/^transformer_encoder_18/StatefulPartitionedCall/^transformer_encoder_19/StatefulPartitionedCall/^transformer_encoder_20/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:���������P	:���������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2P
&FinalLayerNorm/StatefulPartitionedCall&FinalLayerNorm/StatefulPartitionedCall2l
4FullyConnectedLayerAreaRatio/StatefulPartitionedCall4FullyConnectedLayerAreaRatio/StatefulPartitionedCall2Z
+PredictionAreaRatio/StatefulPartitionedCall+PredictionAreaRatio/StatefulPartitionedCall2`
.transformer_encoder_18/StatefulPartitionedCall.transformer_encoder_18/StatefulPartitionedCall2`
.transformer_encoder_19/StatefulPartitionedCall.transformer_encoder_19/StatefulPartitionedCall2`
.transformer_encoder_20/StatefulPartitionedCall.transformer_encoder_20/StatefulPartitionedCall:'7#
!
_user_specified_name	1472976:'6#
!
_user_specified_name	1472974:'5#
!
_user_specified_name	1472960:'4#
!
_user_specified_name	1472958:'3#
!
_user_specified_name	1472910:'2#
!
_user_specified_name	1472908:'1#
!
_user_specified_name	1472862:'0#
!
_user_specified_name	1472860:'/#
!
_user_specified_name	1472858:'.#
!
_user_specified_name	1472856:'-#
!
_user_specified_name	1472854:',#
!
_user_specified_name	1472852:'+#
!
_user_specified_name	1472850:'*#
!
_user_specified_name	1472848:')#
!
_user_specified_name	1472846:'(#
!
_user_specified_name	1472844:''#
!
_user_specified_name	1472842:'&#
!
_user_specified_name	1472840:'%#
!
_user_specified_name	1472838:'$#
!
_user_specified_name	1472836:'##
!
_user_specified_name	1472834:'"#
!
_user_specified_name	1472832:'!#
!
_user_specified_name	1472640:' #
!
_user_specified_name	1472638:'#
!
_user_specified_name	1472636:'#
!
_user_specified_name	1472634:'#
!
_user_specified_name	1472632:'#
!
_user_specified_name	1472630:'#
!
_user_specified_name	1472628:'#
!
_user_specified_name	1472626:'#
!
_user_specified_name	1472624:'#
!
_user_specified_name	1472622:'#
!
_user_specified_name	1472620:'#
!
_user_specified_name	1472618:'#
!
_user_specified_name	1472616:'#
!
_user_specified_name	1472614:'#
!
_user_specified_name	1472612:'#
!
_user_specified_name	1472610:'#
!
_user_specified_name	1472418:'#
!
_user_specified_name	1472416:'#
!
_user_specified_name	1472414:'#
!
_user_specified_name	1472412:'#
!
_user_specified_name	1472410:'#
!
_user_specified_name	1472408:'#
!
_user_specified_name	1472406:'
#
!
_user_specified_name	1472404:'	#
!
_user_specified_name	1472402:'#
!
_user_specified_name	1472400:'#
!
_user_specified_name	1472398:'#
!
_user_specified_name	1472396:'#
!
_user_specified_name	1472394:'#
!
_user_specified_name	1472392:'#
!
_user_specified_name	1472390:'#
!
_user_specified_name	1472388:WS
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
�
�
5__inference_PredictionAreaRatio_layer_call_fn_1475791

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
P__inference_PredictionAreaRatio_layer_call_and_return_conditional_losses_1472973o
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
_user_specified_name	1475787:'#
!
_user_specified_name	1475785:O K
'
_output_shapes
:���������

 
_user_specified_nameinputs
�
y
]__inference_ReduceStackDimensionViaSummation_layer_call_and_return_conditional_losses_1475716

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
6__inference_StandardizeTimeLimit_layer_call_fn_1475726

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
Q__inference_StandardizeTimeLimit_layer_call_and_return_conditional_losses_1473638`
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
�,
�
)__inference_model_6_layer_call_fn_1473886
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
GPU2*0J 8� *M
fHRF
D__inference_model_6_layer_call_and_return_conditional_losses_1473658`
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
StatefulPartitionedCallStatefulPartitionedCall:'7#
!
_user_specified_name	1473882:'6#
!
_user_specified_name	1473880:'5#
!
_user_specified_name	1473878:'4#
!
_user_specified_name	1473876:'3#
!
_user_specified_name	1473874:'2#
!
_user_specified_name	1473872:'1#
!
_user_specified_name	1473870:'0#
!
_user_specified_name	1473868:'/#
!
_user_specified_name	1473866:'.#
!
_user_specified_name	1473864:'-#
!
_user_specified_name	1473862:',#
!
_user_specified_name	1473860:'+#
!
_user_specified_name	1473858:'*#
!
_user_specified_name	1473856:')#
!
_user_specified_name	1473854:'(#
!
_user_specified_name	1473852:''#
!
_user_specified_name	1473850:'&#
!
_user_specified_name	1473848:'%#
!
_user_specified_name	1473846:'$#
!
_user_specified_name	1473844:'##
!
_user_specified_name	1473842:'"#
!
_user_specified_name	1473840:'!#
!
_user_specified_name	1473838:' #
!
_user_specified_name	1473836:'#
!
_user_specified_name	1473834:'#
!
_user_specified_name	1473832:'#
!
_user_specified_name	1473830:'#
!
_user_specified_name	1473828:'#
!
_user_specified_name	1473826:'#
!
_user_specified_name	1473824:'#
!
_user_specified_name	1473822:'#
!
_user_specified_name	1473820:'#
!
_user_specified_name	1473818:'#
!
_user_specified_name	1473816:'#
!
_user_specified_name	1473814:'#
!
_user_specified_name	1473812:'#
!
_user_specified_name	1473810:'#
!
_user_specified_name	1473808:'#
!
_user_specified_name	1473806:'#
!
_user_specified_name	1473804:'#
!
_user_specified_name	1473802:'#
!
_user_specified_name	1473800:'#
!
_user_specified_name	1473798:'#
!
_user_specified_name	1473796:'#
!
_user_specified_name	1473794:'
#
!
_user_specified_name	1473792:'	#
!
_user_specified_name	1473790:'#
!
_user_specified_name	1473788:'#
!
_user_specified_name	1473786:'#
!
_user_specified_name	1473784:'#
!
_user_specified_name	1473782:'#
!
_user_specified_name	1473780:'#
!
_user_specified_name	1473778:'#
!
_user_specified_name	1473776:WS
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
�
�
0__inference_FinalLayerNorm_layer_call_fn_1475648

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
K__inference_FinalLayerNorm_layer_call_and_return_conditional_losses_1472907s
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
_user_specified_name	1475644:'#
!
_user_specified_name	1475642:S O
+
_output_shapes
:���������P	
 
_user_specified_nameinputs
�
�
8__inference_transformer_encoder_18_layer_call_fn_1474358

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
S__inference_transformer_encoder_18_layer_call_and_return_conditional_losses_1472387s
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
_user_specified_name	1474352:'#
!
_user_specified_name	1474350:'#
!
_user_specified_name	1474348:'#
!
_user_specified_name	1474346:'#
!
_user_specified_name	1474344:'#
!
_user_specified_name	1474342:'
#
!
_user_specified_name	1474340:'	#
!
_user_specified_name	1474338:'#
!
_user_specified_name	1474336:'#
!
_user_specified_name	1474334:'#
!
_user_specified_name	1474332:'#
!
_user_specified_name	1474330:'#
!
_user_specified_name	1474328:'#
!
_user_specified_name	1474326:'#
!
_user_specified_name	1474324:'#
!
_user_specified_name	1474322:S O
+
_output_shapes
:���������P	
 
_user_specified_nameinputs
�+
�
%__inference_signature_wrapper_1474303
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
GPU2*0J 8� *+
f&R$
"__inference__wrapped_model_1472183`
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
StatefulPartitionedCallStatefulPartitionedCall:'7#
!
_user_specified_name	1474299:'6#
!
_user_specified_name	1474297:'5#
!
_user_specified_name	1474295:'4#
!
_user_specified_name	1474293:'3#
!
_user_specified_name	1474291:'2#
!
_user_specified_name	1474289:'1#
!
_user_specified_name	1474287:'0#
!
_user_specified_name	1474285:'/#
!
_user_specified_name	1474283:'.#
!
_user_specified_name	1474281:'-#
!
_user_specified_name	1474279:',#
!
_user_specified_name	1474277:'+#
!
_user_specified_name	1474275:'*#
!
_user_specified_name	1474273:')#
!
_user_specified_name	1474271:'(#
!
_user_specified_name	1474269:''#
!
_user_specified_name	1474267:'&#
!
_user_specified_name	1474265:'%#
!
_user_specified_name	1474263:'$#
!
_user_specified_name	1474261:'##
!
_user_specified_name	1474259:'"#
!
_user_specified_name	1474257:'!#
!
_user_specified_name	1474255:' #
!
_user_specified_name	1474253:'#
!
_user_specified_name	1474251:'#
!
_user_specified_name	1474249:'#
!
_user_specified_name	1474247:'#
!
_user_specified_name	1474245:'#
!
_user_specified_name	1474243:'#
!
_user_specified_name	1474241:'#
!
_user_specified_name	1474239:'#
!
_user_specified_name	1474237:'#
!
_user_specified_name	1474235:'#
!
_user_specified_name	1474233:'#
!
_user_specified_name	1474231:'#
!
_user_specified_name	1474229:'#
!
_user_specified_name	1474227:'#
!
_user_specified_name	1474225:'#
!
_user_specified_name	1474223:'#
!
_user_specified_name	1474221:'#
!
_user_specified_name	1474219:'#
!
_user_specified_name	1474217:'#
!
_user_specified_name	1474215:'#
!
_user_specified_name	1474213:'#
!
_user_specified_name	1474211:'
#
!
_user_specified_name	1474209:'	#
!
_user_specified_name	1474207:'#
!
_user_specified_name	1474205:'#
!
_user_specified_name	1474203:'#
!
_user_specified_name	1474201:'#
!
_user_specified_name	1474199:'#
!
_user_specified_name	1474197:'#
!
_user_specified_name	1474195:'#
!
_user_specified_name	1474193:WS
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
y
M__inference_ConcatenateLayer_layer_call_and_return_conditional_losses_1475755
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
�
�
8__inference_transformer_encoder_19_layer_call_fn_1474798

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
S__inference_transformer_encoder_19_layer_call_and_return_conditional_losses_1472609s
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
_user_specified_name	1474792:'#
!
_user_specified_name	1474790:'#
!
_user_specified_name	1474788:'#
!
_user_specified_name	1474786:'#
!
_user_specified_name	1474784:'#
!
_user_specified_name	1474782:'
#
!
_user_specified_name	1474780:'	#
!
_user_specified_name	1474778:'#
!
_user_specified_name	1474776:'#
!
_user_specified_name	1474774:'#
!
_user_specified_name	1474772:'#
!
_user_specified_name	1474770:'#
!
_user_specified_name	1474768:'#
!
_user_specified_name	1474766:'#
!
_user_specified_name	1474764:'#
!
_user_specified_name	1474762:S O
+
_output_shapes
:���������P	
 
_user_specified_nameinputs
�
_
C__inference_Output_layer_call_and_return_conditional_losses_1472983

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
S__inference_transformer_encoder_18_layer_call_and_return_conditional_losses_1474759

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
dropout_18/IdentityIdentity-Encoder-FeedForwardLayer_2_1/BiasAdd:output:0*
T0*+
_output_shapes
:���������P	�
Encoder-2nd-AdditionLayer-1/addAddV2#Encoder-1st-AdditionLayer-1/add:z:0dropout_18/Identity:output:0*
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
R
6__inference_StandardizeTimeLimit_layer_call_fn_1475721

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
Q__inference_StandardizeTimeLimit_layer_call_and_return_conditional_losses_1472930`
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
��
�
S__inference_transformer_encoder_20_layer_call_and_return_conditional_losses_1472831

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
dropout_20/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?�
dropout_20/dropout/MulMul-Encoder-FeedForwardLayer_2_3/BiasAdd:output:0!dropout_20/dropout/Const:output:0*
T0*+
_output_shapes
:���������P	�
dropout_20/dropout/ShapeShape-Encoder-FeedForwardLayer_2_3/BiasAdd:output:0*
T0*
_output_shapes
::���
/dropout_20/dropout/random_uniform/RandomUniformRandomUniform!dropout_20/dropout/Shape:output:0*
T0*+
_output_shapes
:���������P	*
dtype0*
seed2*
seed��f
!dropout_20/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=�
dropout_20/dropout/GreaterEqualGreaterEqual8dropout_20/dropout/random_uniform/RandomUniform:output:0*dropout_20/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:���������P	_
dropout_20/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout_20/dropout/SelectV2SelectV2#dropout_20/dropout/GreaterEqual:z:0dropout_20/dropout/Mul:z:0#dropout_20/dropout/Const_1:output:0*
T0*+
_output_shapes
:���������P	�
Encoder-2nd-AdditionLayer-3/addAddV2#Encoder-1st-AdditionLayer-3/add:z:0$dropout_20/dropout/SelectV2:output:0*
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
�
_
C__inference_Output_layer_call_and_return_conditional_losses_1475817

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
S__inference_transformer_encoder_20_layer_call_and_return_conditional_losses_1473581

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
dropout_20/IdentityIdentity-Encoder-FeedForwardLayer_2_3/BiasAdd:output:0*
T0*+
_output_shapes
:���������P	�
Encoder-2nd-AdditionLayer-3/addAddV2#Encoder-1st-AdditionLayer-3/add:z:0dropout_20/Identity:output:0*
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
]__inference_ReduceStackDimensionViaSummation_layer_call_and_return_conditional_losses_1472920

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
�g
�
D__inference_model_6_layer_call_and_return_conditional_losses_1473658
stacklevelinputfeatures
timelimitinput,
transformer_encoder_18_1473166:	,
transformer_encoder_18_1473168:	4
transformer_encoder_18_1473170:	0
transformer_encoder_18_1473172:4
transformer_encoder_18_1473174:	0
transformer_encoder_18_1473176:4
transformer_encoder_18_1473178:	0
transformer_encoder_18_1473180:4
transformer_encoder_18_1473182:	,
transformer_encoder_18_1473184:	,
transformer_encoder_18_1473186:	,
transformer_encoder_18_1473188:	0
transformer_encoder_18_1473190:		,
transformer_encoder_18_1473192:	0
transformer_encoder_18_1473194:		,
transformer_encoder_18_1473196:	,
transformer_encoder_19_1473374:	,
transformer_encoder_19_1473376:	4
transformer_encoder_19_1473378:	0
transformer_encoder_19_1473380:4
transformer_encoder_19_1473382:	0
transformer_encoder_19_1473384:4
transformer_encoder_19_1473386:	0
transformer_encoder_19_1473388:4
transformer_encoder_19_1473390:	,
transformer_encoder_19_1473392:	,
transformer_encoder_19_1473394:	,
transformer_encoder_19_1473396:	0
transformer_encoder_19_1473398:		,
transformer_encoder_19_1473400:	0
transformer_encoder_19_1473402:		,
transformer_encoder_19_1473404:	,
transformer_encoder_20_1473582:	,
transformer_encoder_20_1473584:	4
transformer_encoder_20_1473586:	0
transformer_encoder_20_1473588:4
transformer_encoder_20_1473590:	0
transformer_encoder_20_1473592:4
transformer_encoder_20_1473594:	0
transformer_encoder_20_1473596:4
transformer_encoder_20_1473598:	,
transformer_encoder_20_1473600:	,
transformer_encoder_20_1473602:	,
transformer_encoder_20_1473604:	0
transformer_encoder_20_1473606:		,
transformer_encoder_20_1473608:	0
transformer_encoder_20_1473610:		,
transformer_encoder_20_1473612:	$
finallayernorm_1473616:	$
finallayernorm_1473618:	6
$fullyconnectedlayerarearatio_1473641:

2
$fullyconnectedlayerarearatio_1473643:
-
predictionarearatio_1473646:
)
predictionarearatio_1473648:
identity��&FinalLayerNorm/StatefulPartitionedCall�4FullyConnectedLayerAreaRatio/StatefulPartitionedCall�+PredictionAreaRatio/StatefulPartitionedCall�.transformer_encoder_18/StatefulPartitionedCall�.transformer_encoder_19/StatefulPartitionedCall�.transformer_encoder_20/StatefulPartitionedCall�
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
I__inference_MaskingLayer_layer_call_and_return_conditional_losses_1472197�
transformer_encoder_18/CastCast%MaskingLayer/PartitionedCall:output:0*

DstT0*

SrcT0*+
_output_shapes
:���������P	�
.transformer_encoder_18/StatefulPartitionedCallStatefulPartitionedCalltransformer_encoder_18/Cast:y:0transformer_encoder_18_1473166transformer_encoder_18_1473168transformer_encoder_18_1473170transformer_encoder_18_1473172transformer_encoder_18_1473174transformer_encoder_18_1473176transformer_encoder_18_1473178transformer_encoder_18_1473180transformer_encoder_18_1473182transformer_encoder_18_1473184transformer_encoder_18_1473186transformer_encoder_18_1473188transformer_encoder_18_1473190transformer_encoder_18_1473192transformer_encoder_18_1473194transformer_encoder_18_1473196*
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
S__inference_transformer_encoder_18_layer_call_and_return_conditional_losses_1473165�
.transformer_encoder_19/StatefulPartitionedCallStatefulPartitionedCall7transformer_encoder_18/StatefulPartitionedCall:output:0transformer_encoder_19_1473374transformer_encoder_19_1473376transformer_encoder_19_1473378transformer_encoder_19_1473380transformer_encoder_19_1473382transformer_encoder_19_1473384transformer_encoder_19_1473386transformer_encoder_19_1473388transformer_encoder_19_1473390transformer_encoder_19_1473392transformer_encoder_19_1473394transformer_encoder_19_1473396transformer_encoder_19_1473398transformer_encoder_19_1473400transformer_encoder_19_1473402transformer_encoder_19_1473404*
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
S__inference_transformer_encoder_19_layer_call_and_return_conditional_losses_1473373�
.transformer_encoder_20/StatefulPartitionedCallStatefulPartitionedCall7transformer_encoder_19/StatefulPartitionedCall:output:0transformer_encoder_20_1473582transformer_encoder_20_1473584transformer_encoder_20_1473586transformer_encoder_20_1473588transformer_encoder_20_1473590transformer_encoder_20_1473592transformer_encoder_20_1473594transformer_encoder_20_1473596transformer_encoder_20_1473598transformer_encoder_20_1473600transformer_encoder_20_1473602transformer_encoder_20_1473604transformer_encoder_20_1473606transformer_encoder_20_1473608transformer_encoder_20_1473610transformer_encoder_20_1473612*
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
S__inference_transformer_encoder_20_layer_call_and_return_conditional_losses_1473581�
&FinalLayerNorm/StatefulPartitionedCallStatefulPartitionedCall7transformer_encoder_20/StatefulPartitionedCall:output:0finallayernorm_1473616finallayernorm_1473618*
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
K__inference_FinalLayerNorm_layer_call_and_return_conditional_losses_1472907�
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
]__inference_ReduceStackDimensionViaSummation_layer_call_and_return_conditional_losses_1473628r
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
Q__inference_StandardizeTimeLimit_layer_call_and_return_conditional_losses_1473638�
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
M__inference_ConcatenateLayer_layer_call_and_return_conditional_losses_1472938�
4FullyConnectedLayerAreaRatio/StatefulPartitionedCallStatefulPartitionedCall)ConcatenateLayer/PartitionedCall:output:0$fullyconnectedlayerarearatio_1473641$fullyconnectedlayerarearatio_1473643*
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
Y__inference_FullyConnectedLayerAreaRatio_layer_call_and_return_conditional_losses_1472957�
+PredictionAreaRatio/StatefulPartitionedCallStatefulPartitionedCall=FullyConnectedLayerAreaRatio/StatefulPartitionedCall:output:0predictionarearatio_1473646predictionarearatio_1473648*
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
P__inference_PredictionAreaRatio_layer_call_and_return_conditional_losses_1472973�
Output/PartitionedCallPartitionedCall4PredictionAreaRatio/StatefulPartitionedCall:output:0*
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
C__inference_Output_layer_call_and_return_conditional_losses_1473655_
IdentityIdentityOutput/PartitionedCall:output:0^NoOp*
T0*
_output_shapes
:�
NoOpNoOp'^FinalLayerNorm/StatefulPartitionedCall5^FullyConnectedLayerAreaRatio/StatefulPartitionedCall,^PredictionAreaRatio/StatefulPartitionedCall/^transformer_encoder_18/StatefulPartitionedCall/^transformer_encoder_19/StatefulPartitionedCall/^transformer_encoder_20/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:���������P	:���������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2P
&FinalLayerNorm/StatefulPartitionedCall&FinalLayerNorm/StatefulPartitionedCall2l
4FullyConnectedLayerAreaRatio/StatefulPartitionedCall4FullyConnectedLayerAreaRatio/StatefulPartitionedCall2Z
+PredictionAreaRatio/StatefulPartitionedCall+PredictionAreaRatio/StatefulPartitionedCall2`
.transformer_encoder_18/StatefulPartitionedCall.transformer_encoder_18/StatefulPartitionedCall2`
.transformer_encoder_19/StatefulPartitionedCall.transformer_encoder_19/StatefulPartitionedCall2`
.transformer_encoder_20/StatefulPartitionedCall.transformer_encoder_20/StatefulPartitionedCall:'7#
!
_user_specified_name	1473648:'6#
!
_user_specified_name	1473646:'5#
!
_user_specified_name	1473643:'4#
!
_user_specified_name	1473641:'3#
!
_user_specified_name	1473618:'2#
!
_user_specified_name	1473616:'1#
!
_user_specified_name	1473612:'0#
!
_user_specified_name	1473610:'/#
!
_user_specified_name	1473608:'.#
!
_user_specified_name	1473606:'-#
!
_user_specified_name	1473604:',#
!
_user_specified_name	1473602:'+#
!
_user_specified_name	1473600:'*#
!
_user_specified_name	1473598:')#
!
_user_specified_name	1473596:'(#
!
_user_specified_name	1473594:''#
!
_user_specified_name	1473592:'&#
!
_user_specified_name	1473590:'%#
!
_user_specified_name	1473588:'$#
!
_user_specified_name	1473586:'##
!
_user_specified_name	1473584:'"#
!
_user_specified_name	1473582:'!#
!
_user_specified_name	1473404:' #
!
_user_specified_name	1473402:'#
!
_user_specified_name	1473400:'#
!
_user_specified_name	1473398:'#
!
_user_specified_name	1473396:'#
!
_user_specified_name	1473394:'#
!
_user_specified_name	1473392:'#
!
_user_specified_name	1473390:'#
!
_user_specified_name	1473388:'#
!
_user_specified_name	1473386:'#
!
_user_specified_name	1473384:'#
!
_user_specified_name	1473382:'#
!
_user_specified_name	1473380:'#
!
_user_specified_name	1473378:'#
!
_user_specified_name	1473376:'#
!
_user_specified_name	1473374:'#
!
_user_specified_name	1473196:'#
!
_user_specified_name	1473194:'#
!
_user_specified_name	1473192:'#
!
_user_specified_name	1473190:'#
!
_user_specified_name	1473188:'#
!
_user_specified_name	1473186:'#
!
_user_specified_name	1473184:'
#
!
_user_specified_name	1473182:'	#
!
_user_specified_name	1473180:'#
!
_user_specified_name	1473178:'#
!
_user_specified_name	1473176:'#
!
_user_specified_name	1473174:'#
!
_user_specified_name	1473172:'#
!
_user_specified_name	1473170:'#
!
_user_specified_name	1473168:'#
!
_user_specified_name	1473166:WS
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
Q__inference_StandardizeTimeLimit_layer_call_and_return_conditional_losses_1475742

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
�
�
8__inference_transformer_encoder_18_layer_call_fn_1474397

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
S__inference_transformer_encoder_18_layer_call_and_return_conditional_losses_1473165s
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
_user_specified_name	1474391:'#
!
_user_specified_name	1474389:'#
!
_user_specified_name	1474387:'#
!
_user_specified_name	1474385:'#
!
_user_specified_name	1474383:'#
!
_user_specified_name	1474381:'
#
!
_user_specified_name	1474379:'	#
!
_user_specified_name	1474377:'#
!
_user_specified_name	1474375:'#
!
_user_specified_name	1474373:'#
!
_user_specified_name	1474371:'#
!
_user_specified_name	1474369:'#
!
_user_specified_name	1474367:'#
!
_user_specified_name	1474365:'#
!
_user_specified_name	1474363:'#
!
_user_specified_name	1474361:S O
+
_output_shapes
:���������P	
 
_user_specified_nameinputs
��
�
S__inference_transformer_encoder_18_layer_call_and_return_conditional_losses_1472387

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
dropout_18/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?�
dropout_18/dropout/MulMul-Encoder-FeedForwardLayer_2_1/BiasAdd:output:0!dropout_18/dropout/Const:output:0*
T0*+
_output_shapes
:���������P	�
dropout_18/dropout/ShapeShape-Encoder-FeedForwardLayer_2_1/BiasAdd:output:0*
T0*
_output_shapes
::���
/dropout_18/dropout/random_uniform/RandomUniformRandomUniform!dropout_18/dropout/Shape:output:0*
T0*+
_output_shapes
:���������P	*
dtype0*
seed2*
seed��f
!dropout_18/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=�
dropout_18/dropout/GreaterEqualGreaterEqual8dropout_18/dropout/random_uniform/RandomUniform:output:0*dropout_18/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:���������P	_
dropout_18/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout_18/dropout/SelectV2SelectV2#dropout_18/dropout/GreaterEqual:z:0dropout_18/dropout/Mul:z:0#dropout_18/dropout/Const_1:output:0*
T0*+
_output_shapes
:���������P	�
Encoder-2nd-AdditionLayer-1/addAddV2#Encoder-1st-AdditionLayer-1/add:z:0$dropout_18/dropout/SelectV2:output:0*
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
e
I__inference_MaskingLayer_layer_call_and_return_conditional_losses_1474319

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
�
>__inference_FullyConnectedLayerAreaRatio_layer_call_fn_1475764

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
Y__inference_FullyConnectedLayerAreaRatio_layer_call_and_return_conditional_losses_1472957o
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
_user_specified_name	1475760:'#
!
_user_specified_name	1475758:O K
'
_output_shapes
:���������

 
_user_specified_nameinputs
� 
�
K__inference_FinalLayerNorm_layer_call_and_return_conditional_losses_1472907

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
��
�
S__inference_transformer_encoder_18_layer_call_and_return_conditional_losses_1474585

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
dropout_18/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?�
dropout_18/dropout/MulMul-Encoder-FeedForwardLayer_2_1/BiasAdd:output:0!dropout_18/dropout/Const:output:0*
T0*+
_output_shapes
:���������P	�
dropout_18/dropout/ShapeShape-Encoder-FeedForwardLayer_2_1/BiasAdd:output:0*
T0*
_output_shapes
::���
/dropout_18/dropout/random_uniform/RandomUniformRandomUniform!dropout_18/dropout/Shape:output:0*
T0*+
_output_shapes
:���������P	*
dtype0*
seed2*
seed��f
!dropout_18/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=�
dropout_18/dropout/GreaterEqualGreaterEqual8dropout_18/dropout/random_uniform/RandomUniform:output:0*dropout_18/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:���������P	_
dropout_18/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout_18/dropout/SelectV2SelectV2#dropout_18/dropout/GreaterEqual:z:0dropout_18/dropout/Mul:z:0#dropout_18/dropout/Const_1:output:0*
T0*+
_output_shapes
:���������P	�
Encoder-2nd-AdditionLayer-1/addAddV2#Encoder-1st-AdditionLayer-1/add:z:0$dropout_18/dropout/SelectV2:output:0*
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
�
�
Y__inference_FullyConnectedLayerAreaRatio_layer_call_and_return_conditional_losses_1472957

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
� 
�
K__inference_FinalLayerNorm_layer_call_and_return_conditional_losses_1475690

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
�
e
I__inference_MaskingLayer_layer_call_and_return_conditional_losses_1472197

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
�
m
Q__inference_StandardizeTimeLimit_layer_call_and_return_conditional_losses_1473638

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
S__inference_transformer_encoder_20_layer_call_and_return_conditional_losses_1475465

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
dropout_20/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?�
dropout_20/dropout/MulMul-Encoder-FeedForwardLayer_2_3/BiasAdd:output:0!dropout_20/dropout/Const:output:0*
T0*+
_output_shapes
:���������P	�
dropout_20/dropout/ShapeShape-Encoder-FeedForwardLayer_2_3/BiasAdd:output:0*
T0*
_output_shapes
::���
/dropout_20/dropout/random_uniform/RandomUniformRandomUniform!dropout_20/dropout/Shape:output:0*
T0*+
_output_shapes
:���������P	*
dtype0*
seed2*
seed��f
!dropout_20/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=�
dropout_20/dropout/GreaterEqualGreaterEqual8dropout_20/dropout/random_uniform/RandomUniform:output:0*dropout_20/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:���������P	_
dropout_20/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout_20/dropout/SelectV2SelectV2#dropout_20/dropout/GreaterEqual:z:0dropout_20/dropout/Mul:z:0#dropout_20/dropout/Const_1:output:0*
T0*+
_output_shapes
:���������P	�
Encoder-2nd-AdditionLayer-3/addAddV2#Encoder-1st-AdditionLayer-3/add:z:0$dropout_20/dropout/SelectV2:output:0*
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
8__inference_transformer_encoder_20_layer_call_fn_1475238

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
S__inference_transformer_encoder_20_layer_call_and_return_conditional_losses_1472831s
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
_user_specified_name	1475232:'#
!
_user_specified_name	1475230:'#
!
_user_specified_name	1475228:'#
!
_user_specified_name	1475226:'#
!
_user_specified_name	1475224:'#
!
_user_specified_name	1475222:'
#
!
_user_specified_name	1475220:'	#
!
_user_specified_name	1475218:'#
!
_user_specified_name	1475216:'#
!
_user_specified_name	1475214:'#
!
_user_specified_name	1475212:'#
!
_user_specified_name	1475210:'#
!
_user_specified_name	1475208:'#
!
_user_specified_name	1475206:'#
!
_user_specified_name	1475204:'#
!
_user_specified_name	1475202:S O
+
_output_shapes
:���������P	
 
_user_specified_nameinputs
�
y
]__inference_ReduceStackDimensionViaSummation_layer_call_and_return_conditional_losses_1475708

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
�

�
P__inference_PredictionAreaRatio_layer_call_and_return_conditional_losses_1475802

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
S__inference_transformer_encoder_19_layer_call_and_return_conditional_losses_1475025

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
dropout_19/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?�
dropout_19/dropout/MulMul-Encoder-FeedForwardLayer_2_2/BiasAdd:output:0!dropout_19/dropout/Const:output:0*
T0*+
_output_shapes
:���������P	�
dropout_19/dropout/ShapeShape-Encoder-FeedForwardLayer_2_2/BiasAdd:output:0*
T0*
_output_shapes
::���
/dropout_19/dropout/random_uniform/RandomUniformRandomUniform!dropout_19/dropout/Shape:output:0*
T0*+
_output_shapes
:���������P	*
dtype0*
seed2*
seed��f
!dropout_19/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=�
dropout_19/dropout/GreaterEqualGreaterEqual8dropout_19/dropout/random_uniform/RandomUniform:output:0*dropout_19/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:���������P	_
dropout_19/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout_19/dropout/SelectV2SelectV2#dropout_19/dropout/GreaterEqual:z:0dropout_19/dropout/Mul:z:0#dropout_19/dropout/Const_1:output:0*
T0*+
_output_shapes
:���������P	�
Encoder-2nd-AdditionLayer-2/addAddV2#Encoder-1st-AdditionLayer-2/add:z:0$dropout_19/dropout/SelectV2:output:0*
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
�
D
(__inference_Output_layer_call_fn_1475807

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
C__inference_Output_layer_call_and_return_conditional_losses_1472983Q
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
�
J
.__inference_MaskingLayer_layer_call_fn_1474308

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
I__inference_MaskingLayer_layer_call_and_return_conditional_losses_1472197d
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
�
�
8__inference_transformer_encoder_19_layer_call_fn_1474837

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
S__inference_transformer_encoder_19_layer_call_and_return_conditional_losses_1473373s
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
_user_specified_name	1474831:'#
!
_user_specified_name	1474829:'#
!
_user_specified_name	1474827:'#
!
_user_specified_name	1474825:'#
!
_user_specified_name	1474823:'#
!
_user_specified_name	1474821:'
#
!
_user_specified_name	1474819:'	#
!
_user_specified_name	1474817:'#
!
_user_specified_name	1474815:'#
!
_user_specified_name	1474813:'#
!
_user_specified_name	1474811:'#
!
_user_specified_name	1474809:'#
!
_user_specified_name	1474807:'#
!
_user_specified_name	1474805:'#
!
_user_specified_name	1474803:'#
!
_user_specified_name	1474801:S O
+
_output_shapes
:���������P	
 
_user_specified_nameinputs
�
�
8__inference_transformer_encoder_20_layer_call_fn_1475277

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
S__inference_transformer_encoder_20_layer_call_and_return_conditional_losses_1473581s
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
_user_specified_name	1475271:'#
!
_user_specified_name	1475269:'#
!
_user_specified_name	1475267:'#
!
_user_specified_name	1475265:'#
!
_user_specified_name	1475263:'#
!
_user_specified_name	1475261:'
#
!
_user_specified_name	1475259:'	#
!
_user_specified_name	1475257:'#
!
_user_specified_name	1475255:'#
!
_user_specified_name	1475253:'#
!
_user_specified_name	1475251:'#
!
_user_specified_name	1475249:'#
!
_user_specified_name	1475247:'#
!
_user_specified_name	1475245:'#
!
_user_specified_name	1475243:'#
!
_user_specified_name	1475241:S O
+
_output_shapes
:���������P	
 
_user_specified_nameinputs
�
�
Y__inference_FullyConnectedLayerAreaRatio_layer_call_and_return_conditional_losses_1475782

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
B__inference_ReduceStackDimensionViaSummation_layer_call_fn_1475695

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
]__inference_ReduceStackDimensionViaSummation_layer_call_and_return_conditional_losses_1472920`
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
w
M__inference_ConcatenateLayer_layer_call_and_return_conditional_losses_1472938

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
S__inference_transformer_encoder_19_layer_call_and_return_conditional_losses_1472609

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
dropout_19/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?�
dropout_19/dropout/MulMul-Encoder-FeedForwardLayer_2_2/BiasAdd:output:0!dropout_19/dropout/Const:output:0*
T0*+
_output_shapes
:���������P	�
dropout_19/dropout/ShapeShape-Encoder-FeedForwardLayer_2_2/BiasAdd:output:0*
T0*
_output_shapes
::���
/dropout_19/dropout/random_uniform/RandomUniformRandomUniform!dropout_19/dropout/Shape:output:0*
T0*+
_output_shapes
:���������P	*
dtype0*
seed2*
seed��f
!dropout_19/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=�
dropout_19/dropout/GreaterEqualGreaterEqual8dropout_19/dropout/random_uniform/RandomUniform:output:0*dropout_19/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:���������P	_
dropout_19/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout_19/dropout/SelectV2SelectV2#dropout_19/dropout/GreaterEqual:z:0dropout_19/dropout/Mul:z:0#dropout_19/dropout/Const_1:output:0*
T0*+
_output_shapes
:���������P	�
Encoder-2nd-AdditionLayer-2/addAddV2#Encoder-1st-AdditionLayer-2/add:z:0$dropout_19/dropout/SelectV2:output:0*
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
S__inference_transformer_encoder_19_layer_call_and_return_conditional_losses_1473373

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
dropout_19/IdentityIdentity-Encoder-FeedForwardLayer_2_2/BiasAdd:output:0*
T0*+
_output_shapes
:���������P	�
Encoder-2nd-AdditionLayer-2/addAddV2#Encoder-1st-AdditionLayer-2/add:z:0dropout_19/Identity:output:0*
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
_
C__inference_Output_layer_call_and_return_conditional_losses_1475822

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
�
_
C__inference_Output_layer_call_and_return_conditional_losses_1473655

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
D
(__inference_Output_layer_call_fn_1475812

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
C__inference_Output_layer_call_and_return_conditional_losses_1473655Q
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
�
m
Q__inference_StandardizeTimeLimit_layer_call_and_return_conditional_losses_1475734

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
�2
#__inference__traced_restore_1476340
file_prefix3
%assignvariableop_finallayernorm_gamma:	4
&assignvariableop_1_finallayernorm_beta:	H
6assignvariableop_2_fullyconnectedlayerarearatio_kernel:

B
4assignvariableop_3_fullyconnectedlayerarearatio_bias:
?
-assignvariableop_4_predictionarearatio_kernel:
9
+assignvariableop_5_predictionarearatio_bias:i
Sassignvariableop_6_transformer_encoder_18_encoder_selfattentionlayer_1_query_kernel:	c
Qassignvariableop_7_transformer_encoder_18_encoder_selfattentionlayer_1_query_bias:g
Qassignvariableop_8_transformer_encoder_18_encoder_selfattentionlayer_1_key_kernel:	a
Oassignvariableop_9_transformer_encoder_18_encoder_selfattentionlayer_1_key_bias:j
Tassignvariableop_10_transformer_encoder_18_encoder_selfattentionlayer_1_value_kernel:	d
Rassignvariableop_11_transformer_encoder_18_encoder_selfattentionlayer_1_value_bias:u
_assignvariableop_12_transformer_encoder_18_encoder_selfattentionlayer_1_attention_output_kernel:	k
]assignvariableop_13_transformer_encoder_18_encoder_selfattentionlayer_1_attention_output_bias:	_
Qassignvariableop_14_transformer_encoder_18_encoder_1st_normalizationlayer_1_gamma:	^
Passignvariableop_15_transformer_encoder_18_encoder_1st_normalizationlayer_1_beta:	_
Qassignvariableop_16_transformer_encoder_18_encoder_2nd_normalizationlayer_1_gamma:	^
Passignvariableop_17_transformer_encoder_18_encoder_2nd_normalizationlayer_1_beta:	`
Nassignvariableop_18_transformer_encoder_18_encoder_feedforwardlayer_1_1_kernel:		Z
Lassignvariableop_19_transformer_encoder_18_encoder_feedforwardlayer_1_1_bias:	`
Nassignvariableop_20_transformer_encoder_18_encoder_feedforwardlayer_2_1_kernel:		Z
Lassignvariableop_21_transformer_encoder_18_encoder_feedforwardlayer_2_1_bias:	j
Tassignvariableop_22_transformer_encoder_19_encoder_selfattentionlayer_2_query_kernel:	d
Rassignvariableop_23_transformer_encoder_19_encoder_selfattentionlayer_2_query_bias:h
Rassignvariableop_24_transformer_encoder_19_encoder_selfattentionlayer_2_key_kernel:	b
Passignvariableop_25_transformer_encoder_19_encoder_selfattentionlayer_2_key_bias:j
Tassignvariableop_26_transformer_encoder_19_encoder_selfattentionlayer_2_value_kernel:	d
Rassignvariableop_27_transformer_encoder_19_encoder_selfattentionlayer_2_value_bias:u
_assignvariableop_28_transformer_encoder_19_encoder_selfattentionlayer_2_attention_output_kernel:	k
]assignvariableop_29_transformer_encoder_19_encoder_selfattentionlayer_2_attention_output_bias:	_
Qassignvariableop_30_transformer_encoder_19_encoder_1st_normalizationlayer_2_gamma:	^
Passignvariableop_31_transformer_encoder_19_encoder_1st_normalizationlayer_2_beta:	_
Qassignvariableop_32_transformer_encoder_19_encoder_2nd_normalizationlayer_2_gamma:	^
Passignvariableop_33_transformer_encoder_19_encoder_2nd_normalizationlayer_2_beta:	`
Nassignvariableop_34_transformer_encoder_19_encoder_feedforwardlayer_1_2_kernel:		Z
Lassignvariableop_35_transformer_encoder_19_encoder_feedforwardlayer_1_2_bias:	`
Nassignvariableop_36_transformer_encoder_19_encoder_feedforwardlayer_2_2_kernel:		Z
Lassignvariableop_37_transformer_encoder_19_encoder_feedforwardlayer_2_2_bias:	j
Tassignvariableop_38_transformer_encoder_20_encoder_selfattentionlayer_3_query_kernel:	d
Rassignvariableop_39_transformer_encoder_20_encoder_selfattentionlayer_3_query_bias:h
Rassignvariableop_40_transformer_encoder_20_encoder_selfattentionlayer_3_key_kernel:	b
Passignvariableop_41_transformer_encoder_20_encoder_selfattentionlayer_3_key_bias:j
Tassignvariableop_42_transformer_encoder_20_encoder_selfattentionlayer_3_value_kernel:	d
Rassignvariableop_43_transformer_encoder_20_encoder_selfattentionlayer_3_value_bias:u
_assignvariableop_44_transformer_encoder_20_encoder_selfattentionlayer_3_attention_output_kernel:	k
]assignvariableop_45_transformer_encoder_20_encoder_selfattentionlayer_3_attention_output_bias:	_
Qassignvariableop_46_transformer_encoder_20_encoder_1st_normalizationlayer_3_gamma:	^
Passignvariableop_47_transformer_encoder_20_encoder_1st_normalizationlayer_3_beta:	_
Qassignvariableop_48_transformer_encoder_20_encoder_2nd_normalizationlayer_3_gamma:	^
Passignvariableop_49_transformer_encoder_20_encoder_2nd_normalizationlayer_3_beta:	`
Nassignvariableop_50_transformer_encoder_20_encoder_feedforwardlayer_1_3_kernel:		Z
Lassignvariableop_51_transformer_encoder_20_encoder_feedforwardlayer_1_3_bias:	`
Nassignvariableop_52_transformer_encoder_20_encoder_feedforwardlayer_2_3_kernel:		Z
Lassignvariableop_53_transformer_encoder_20_encoder_feedforwardlayer_2_3_bias:	
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
AssignVariableOp_4AssignVariableOp-assignvariableop_4_predictionarearatio_kernelIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOp+assignvariableop_5_predictionarearatio_biasIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOpSassignvariableop_6_transformer_encoder_18_encoder_selfattentionlayer_1_query_kernelIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOpQassignvariableop_7_transformer_encoder_18_encoder_selfattentionlayer_1_query_biasIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOpQassignvariableop_8_transformer_encoder_18_encoder_selfattentionlayer_1_key_kernelIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOpOassignvariableop_9_transformer_encoder_18_encoder_selfattentionlayer_1_key_biasIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOpTassignvariableop_10_transformer_encoder_18_encoder_selfattentionlayer_1_value_kernelIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOpRassignvariableop_11_transformer_encoder_18_encoder_selfattentionlayer_1_value_biasIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOp_assignvariableop_12_transformer_encoder_18_encoder_selfattentionlayer_1_attention_output_kernelIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOp]assignvariableop_13_transformer_encoder_18_encoder_selfattentionlayer_1_attention_output_biasIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOpQassignvariableop_14_transformer_encoder_18_encoder_1st_normalizationlayer_1_gammaIdentity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOpPassignvariableop_15_transformer_encoder_18_encoder_1st_normalizationlayer_1_betaIdentity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOpQassignvariableop_16_transformer_encoder_18_encoder_2nd_normalizationlayer_1_gammaIdentity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOpPassignvariableop_17_transformer_encoder_18_encoder_2nd_normalizationlayer_1_betaIdentity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_18AssignVariableOpNassignvariableop_18_transformer_encoder_18_encoder_feedforwardlayer_1_1_kernelIdentity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOpLassignvariableop_19_transformer_encoder_18_encoder_feedforwardlayer_1_1_biasIdentity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_20AssignVariableOpNassignvariableop_20_transformer_encoder_18_encoder_feedforwardlayer_2_1_kernelIdentity_20:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_21AssignVariableOpLassignvariableop_21_transformer_encoder_18_encoder_feedforwardlayer_2_1_biasIdentity_21:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_22AssignVariableOpTassignvariableop_22_transformer_encoder_19_encoder_selfattentionlayer_2_query_kernelIdentity_22:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_23AssignVariableOpRassignvariableop_23_transformer_encoder_19_encoder_selfattentionlayer_2_query_biasIdentity_23:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_24AssignVariableOpRassignvariableop_24_transformer_encoder_19_encoder_selfattentionlayer_2_key_kernelIdentity_24:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_25AssignVariableOpPassignvariableop_25_transformer_encoder_19_encoder_selfattentionlayer_2_key_biasIdentity_25:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_26AssignVariableOpTassignvariableop_26_transformer_encoder_19_encoder_selfattentionlayer_2_value_kernelIdentity_26:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_27AssignVariableOpRassignvariableop_27_transformer_encoder_19_encoder_selfattentionlayer_2_value_biasIdentity_27:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_28AssignVariableOp_assignvariableop_28_transformer_encoder_19_encoder_selfattentionlayer_2_attention_output_kernelIdentity_28:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_29AssignVariableOp]assignvariableop_29_transformer_encoder_19_encoder_selfattentionlayer_2_attention_output_biasIdentity_29:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_30AssignVariableOpQassignvariableop_30_transformer_encoder_19_encoder_1st_normalizationlayer_2_gammaIdentity_30:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_31AssignVariableOpPassignvariableop_31_transformer_encoder_19_encoder_1st_normalizationlayer_2_betaIdentity_31:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_32AssignVariableOpQassignvariableop_32_transformer_encoder_19_encoder_2nd_normalizationlayer_2_gammaIdentity_32:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_33AssignVariableOpPassignvariableop_33_transformer_encoder_19_encoder_2nd_normalizationlayer_2_betaIdentity_33:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_34AssignVariableOpNassignvariableop_34_transformer_encoder_19_encoder_feedforwardlayer_1_2_kernelIdentity_34:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_35AssignVariableOpLassignvariableop_35_transformer_encoder_19_encoder_feedforwardlayer_1_2_biasIdentity_35:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_36AssignVariableOpNassignvariableop_36_transformer_encoder_19_encoder_feedforwardlayer_2_2_kernelIdentity_36:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_37AssignVariableOpLassignvariableop_37_transformer_encoder_19_encoder_feedforwardlayer_2_2_biasIdentity_37:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_38AssignVariableOpTassignvariableop_38_transformer_encoder_20_encoder_selfattentionlayer_3_query_kernelIdentity_38:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_39AssignVariableOpRassignvariableop_39_transformer_encoder_20_encoder_selfattentionlayer_3_query_biasIdentity_39:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_40AssignVariableOpRassignvariableop_40_transformer_encoder_20_encoder_selfattentionlayer_3_key_kernelIdentity_40:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_41AssignVariableOpPassignvariableop_41_transformer_encoder_20_encoder_selfattentionlayer_3_key_biasIdentity_41:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_42AssignVariableOpTassignvariableop_42_transformer_encoder_20_encoder_selfattentionlayer_3_value_kernelIdentity_42:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_43AssignVariableOpRassignvariableop_43_transformer_encoder_20_encoder_selfattentionlayer_3_value_biasIdentity_43:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_44AssignVariableOp_assignvariableop_44_transformer_encoder_20_encoder_selfattentionlayer_3_attention_output_kernelIdentity_44:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_45AssignVariableOp]assignvariableop_45_transformer_encoder_20_encoder_selfattentionlayer_3_attention_output_biasIdentity_45:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_46AssignVariableOpQassignvariableop_46_transformer_encoder_20_encoder_1st_normalizationlayer_3_gammaIdentity_46:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_47AssignVariableOpPassignvariableop_47_transformer_encoder_20_encoder_1st_normalizationlayer_3_betaIdentity_47:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_48AssignVariableOpQassignvariableop_48_transformer_encoder_20_encoder_2nd_normalizationlayer_3_gammaIdentity_48:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_49AssignVariableOpPassignvariableop_49_transformer_encoder_20_encoder_2nd_normalizationlayer_3_betaIdentity_49:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_50AssignVariableOpNassignvariableop_50_transformer_encoder_20_encoder_feedforwardlayer_1_3_kernelIdentity_50:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_51AssignVariableOpLassignvariableop_51_transformer_encoder_20_encoder_feedforwardlayer_1_3_biasIdentity_51:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_52AssignVariableOpNassignvariableop_52_transformer_encoder_20_encoder_feedforwardlayer_2_3_kernelIdentity_52:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_53AssignVariableOpLassignvariableop_53_transformer_encoder_20_encoder_feedforwardlayer_2_3_biasIdentity_53:output:0"/device:CPU:0*&
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
AssignVariableOpAssignVariableOp:X6T
R
_user_specified_name:8transformer_encoder_20/Encoder-FeedForwardLayer_2_3/bias:Z5V
T
_user_specified_name<:transformer_encoder_20/Encoder-FeedForwardLayer_2_3/kernel:X4T
R
_user_specified_name:8transformer_encoder_20/Encoder-FeedForwardLayer_1_3/bias:Z3V
T
_user_specified_name<:transformer_encoder_20/Encoder-FeedForwardLayer_1_3/kernel:\2X
V
_user_specified_name><transformer_encoder_20/Encoder-2nd-NormalizationLayer-3/beta:]1Y
W
_user_specified_name?=transformer_encoder_20/Encoder-2nd-NormalizationLayer-3/gamma:\0X
V
_user_specified_name><transformer_encoder_20/Encoder-1st-NormalizationLayer-3/beta:]/Y
W
_user_specified_name?=transformer_encoder_20/Encoder-1st-NormalizationLayer-3/gamma:i.e
c
_user_specified_nameKItransformer_encoder_20/Encoder-SelfAttentionLayer-3/attention_output/bias:k-g
e
_user_specified_nameMKtransformer_encoder_20/Encoder-SelfAttentionLayer-3/attention_output/kernel:^,Z
X
_user_specified_name@>transformer_encoder_20/Encoder-SelfAttentionLayer-3/value/bias:`+\
Z
_user_specified_nameB@transformer_encoder_20/Encoder-SelfAttentionLayer-3/value/kernel:\*X
V
_user_specified_name><transformer_encoder_20/Encoder-SelfAttentionLayer-3/key/bias:^)Z
X
_user_specified_name@>transformer_encoder_20/Encoder-SelfAttentionLayer-3/key/kernel:^(Z
X
_user_specified_name@>transformer_encoder_20/Encoder-SelfAttentionLayer-3/query/bias:`'\
Z
_user_specified_nameB@transformer_encoder_20/Encoder-SelfAttentionLayer-3/query/kernel:X&T
R
_user_specified_name:8transformer_encoder_19/Encoder-FeedForwardLayer_2_2/bias:Z%V
T
_user_specified_name<:transformer_encoder_19/Encoder-FeedForwardLayer_2_2/kernel:X$T
R
_user_specified_name:8transformer_encoder_19/Encoder-FeedForwardLayer_1_2/bias:Z#V
T
_user_specified_name<:transformer_encoder_19/Encoder-FeedForwardLayer_1_2/kernel:\"X
V
_user_specified_name><transformer_encoder_19/Encoder-2nd-NormalizationLayer-2/beta:]!Y
W
_user_specified_name?=transformer_encoder_19/Encoder-2nd-NormalizationLayer-2/gamma:\ X
V
_user_specified_name><transformer_encoder_19/Encoder-1st-NormalizationLayer-2/beta:]Y
W
_user_specified_name?=transformer_encoder_19/Encoder-1st-NormalizationLayer-2/gamma:ie
c
_user_specified_nameKItransformer_encoder_19/Encoder-SelfAttentionLayer-2/attention_output/bias:kg
e
_user_specified_nameMKtransformer_encoder_19/Encoder-SelfAttentionLayer-2/attention_output/kernel:^Z
X
_user_specified_name@>transformer_encoder_19/Encoder-SelfAttentionLayer-2/value/bias:`\
Z
_user_specified_nameB@transformer_encoder_19/Encoder-SelfAttentionLayer-2/value/kernel:\X
V
_user_specified_name><transformer_encoder_19/Encoder-SelfAttentionLayer-2/key/bias:^Z
X
_user_specified_name@>transformer_encoder_19/Encoder-SelfAttentionLayer-2/key/kernel:^Z
X
_user_specified_name@>transformer_encoder_19/Encoder-SelfAttentionLayer-2/query/bias:`\
Z
_user_specified_nameB@transformer_encoder_19/Encoder-SelfAttentionLayer-2/query/kernel:XT
R
_user_specified_name:8transformer_encoder_18/Encoder-FeedForwardLayer_2_1/bias:ZV
T
_user_specified_name<:transformer_encoder_18/Encoder-FeedForwardLayer_2_1/kernel:XT
R
_user_specified_name:8transformer_encoder_18/Encoder-FeedForwardLayer_1_1/bias:ZV
T
_user_specified_name<:transformer_encoder_18/Encoder-FeedForwardLayer_1_1/kernel:\X
V
_user_specified_name><transformer_encoder_18/Encoder-2nd-NormalizationLayer-1/beta:]Y
W
_user_specified_name?=transformer_encoder_18/Encoder-2nd-NormalizationLayer-1/gamma:\X
V
_user_specified_name><transformer_encoder_18/Encoder-1st-NormalizationLayer-1/beta:]Y
W
_user_specified_name?=transformer_encoder_18/Encoder-1st-NormalizationLayer-1/gamma:ie
c
_user_specified_nameKItransformer_encoder_18/Encoder-SelfAttentionLayer-1/attention_output/bias:kg
e
_user_specified_nameMKtransformer_encoder_18/Encoder-SelfAttentionLayer-1/attention_output/kernel:^Z
X
_user_specified_name@>transformer_encoder_18/Encoder-SelfAttentionLayer-1/value/bias:`\
Z
_user_specified_nameB@transformer_encoder_18/Encoder-SelfAttentionLayer-1/value/kernel:\
X
V
_user_specified_name><transformer_encoder_18/Encoder-SelfAttentionLayer-1/key/bias:^	Z
X
_user_specified_name@>transformer_encoder_18/Encoder-SelfAttentionLayer-1/key/kernel:^Z
X
_user_specified_name@>transformer_encoder_18/Encoder-SelfAttentionLayer-1/query/bias:`\
Z
_user_specified_nameB@transformer_encoder_18/Encoder-SelfAttentionLayer-1/query/kernel:84
2
_user_specified_namePredictionAreaRatio/bias::6
4
_user_specified_namePredictionAreaRatio/kernel:A=
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
��
�
S__inference_transformer_encoder_20_layer_call_and_return_conditional_losses_1475639

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
dropout_20/IdentityIdentity-Encoder-FeedForwardLayer_2_3/BiasAdd:output:0*
T0*+
_output_shapes
:���������P	�
Encoder-2nd-AdditionLayer-3/addAddV2#Encoder-1st-AdditionLayer-3/add:z:0dropout_20/Identity:output:0*
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
��
�
S__inference_transformer_encoder_19_layer_call_and_return_conditional_losses_1475199

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
dropout_19/IdentityIdentity-Encoder-FeedForwardLayer_2_2/BiasAdd:output:0*
T0*+
_output_shapes
:���������P	�
Encoder-2nd-AdditionLayer-2/addAddV2#Encoder-1st-AdditionLayer-2/add:z:0dropout_19/Identity:output:0*
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
S__inference_transformer_encoder_18_layer_call_and_return_conditional_losses_1473165

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
dropout_18/IdentityIdentity-Encoder-FeedForwardLayer_2_1/BiasAdd:output:0*
T0*+
_output_shapes
:���������P	�
Encoder-2nd-AdditionLayer-1/addAddV2#Encoder-1st-AdditionLayer-1/add:z:0dropout_18/Identity:output:0*
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
��
�B
 __inference__traced_save_1476169
file_prefix9
+read_disablecopyonread_finallayernorm_gamma:	:
,read_1_disablecopyonread_finallayernorm_beta:	N
<read_2_disablecopyonread_fullyconnectedlayerarearatio_kernel:

H
:read_3_disablecopyonread_fullyconnectedlayerarearatio_bias:
E
3read_4_disablecopyonread_predictionarearatio_kernel:
?
1read_5_disablecopyonread_predictionarearatio_bias:o
Yread_6_disablecopyonread_transformer_encoder_18_encoder_selfattentionlayer_1_query_kernel:	i
Wread_7_disablecopyonread_transformer_encoder_18_encoder_selfattentionlayer_1_query_bias:m
Wread_8_disablecopyonread_transformer_encoder_18_encoder_selfattentionlayer_1_key_kernel:	g
Uread_9_disablecopyonread_transformer_encoder_18_encoder_selfattentionlayer_1_key_bias:p
Zread_10_disablecopyonread_transformer_encoder_18_encoder_selfattentionlayer_1_value_kernel:	j
Xread_11_disablecopyonread_transformer_encoder_18_encoder_selfattentionlayer_1_value_bias:{
eread_12_disablecopyonread_transformer_encoder_18_encoder_selfattentionlayer_1_attention_output_kernel:	q
cread_13_disablecopyonread_transformer_encoder_18_encoder_selfattentionlayer_1_attention_output_bias:	e
Wread_14_disablecopyonread_transformer_encoder_18_encoder_1st_normalizationlayer_1_gamma:	d
Vread_15_disablecopyonread_transformer_encoder_18_encoder_1st_normalizationlayer_1_beta:	e
Wread_16_disablecopyonread_transformer_encoder_18_encoder_2nd_normalizationlayer_1_gamma:	d
Vread_17_disablecopyonread_transformer_encoder_18_encoder_2nd_normalizationlayer_1_beta:	f
Tread_18_disablecopyonread_transformer_encoder_18_encoder_feedforwardlayer_1_1_kernel:		`
Rread_19_disablecopyonread_transformer_encoder_18_encoder_feedforwardlayer_1_1_bias:	f
Tread_20_disablecopyonread_transformer_encoder_18_encoder_feedforwardlayer_2_1_kernel:		`
Rread_21_disablecopyonread_transformer_encoder_18_encoder_feedforwardlayer_2_1_bias:	p
Zread_22_disablecopyonread_transformer_encoder_19_encoder_selfattentionlayer_2_query_kernel:	j
Xread_23_disablecopyonread_transformer_encoder_19_encoder_selfattentionlayer_2_query_bias:n
Xread_24_disablecopyonread_transformer_encoder_19_encoder_selfattentionlayer_2_key_kernel:	h
Vread_25_disablecopyonread_transformer_encoder_19_encoder_selfattentionlayer_2_key_bias:p
Zread_26_disablecopyonread_transformer_encoder_19_encoder_selfattentionlayer_2_value_kernel:	j
Xread_27_disablecopyonread_transformer_encoder_19_encoder_selfattentionlayer_2_value_bias:{
eread_28_disablecopyonread_transformer_encoder_19_encoder_selfattentionlayer_2_attention_output_kernel:	q
cread_29_disablecopyonread_transformer_encoder_19_encoder_selfattentionlayer_2_attention_output_bias:	e
Wread_30_disablecopyonread_transformer_encoder_19_encoder_1st_normalizationlayer_2_gamma:	d
Vread_31_disablecopyonread_transformer_encoder_19_encoder_1st_normalizationlayer_2_beta:	e
Wread_32_disablecopyonread_transformer_encoder_19_encoder_2nd_normalizationlayer_2_gamma:	d
Vread_33_disablecopyonread_transformer_encoder_19_encoder_2nd_normalizationlayer_2_beta:	f
Tread_34_disablecopyonread_transformer_encoder_19_encoder_feedforwardlayer_1_2_kernel:		`
Rread_35_disablecopyonread_transformer_encoder_19_encoder_feedforwardlayer_1_2_bias:	f
Tread_36_disablecopyonread_transformer_encoder_19_encoder_feedforwardlayer_2_2_kernel:		`
Rread_37_disablecopyonread_transformer_encoder_19_encoder_feedforwardlayer_2_2_bias:	p
Zread_38_disablecopyonread_transformer_encoder_20_encoder_selfattentionlayer_3_query_kernel:	j
Xread_39_disablecopyonread_transformer_encoder_20_encoder_selfattentionlayer_3_query_bias:n
Xread_40_disablecopyonread_transformer_encoder_20_encoder_selfattentionlayer_3_key_kernel:	h
Vread_41_disablecopyonread_transformer_encoder_20_encoder_selfattentionlayer_3_key_bias:p
Zread_42_disablecopyonread_transformer_encoder_20_encoder_selfattentionlayer_3_value_kernel:	j
Xread_43_disablecopyonread_transformer_encoder_20_encoder_selfattentionlayer_3_value_bias:{
eread_44_disablecopyonread_transformer_encoder_20_encoder_selfattentionlayer_3_attention_output_kernel:	q
cread_45_disablecopyonread_transformer_encoder_20_encoder_selfattentionlayer_3_attention_output_bias:	e
Wread_46_disablecopyonread_transformer_encoder_20_encoder_1st_normalizationlayer_3_gamma:	d
Vread_47_disablecopyonread_transformer_encoder_20_encoder_1st_normalizationlayer_3_beta:	e
Wread_48_disablecopyonread_transformer_encoder_20_encoder_2nd_normalizationlayer_3_gamma:	d
Vread_49_disablecopyonread_transformer_encoder_20_encoder_2nd_normalizationlayer_3_beta:	f
Tread_50_disablecopyonread_transformer_encoder_20_encoder_feedforwardlayer_1_3_kernel:		`
Rread_51_disablecopyonread_transformer_encoder_20_encoder_feedforwardlayer_1_3_bias:	f
Tread_52_disablecopyonread_transformer_encoder_20_encoder_feedforwardlayer_2_3_kernel:		`
Rread_53_disablecopyonread_transformer_encoder_20_encoder_feedforwardlayer_2_3_bias:	
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
Read_4/DisableCopyOnReadDisableCopyOnRead3read_4_disablecopyonread_predictionarearatio_kernel"/device:CPU:0*
_output_shapes
 �
Read_4/ReadVariableOpReadVariableOp3read_4_disablecopyonread_predictionarearatio_kernel^Read_4/DisableCopyOnRead"/device:CPU:0*
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
Read_5/DisableCopyOnReadDisableCopyOnRead1read_5_disablecopyonread_predictionarearatio_bias"/device:CPU:0*
_output_shapes
 �
Read_5/ReadVariableOpReadVariableOp1read_5_disablecopyonread_predictionarearatio_bias^Read_5/DisableCopyOnRead"/device:CPU:0*
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
Read_6/DisableCopyOnReadDisableCopyOnReadYread_6_disablecopyonread_transformer_encoder_18_encoder_selfattentionlayer_1_query_kernel"/device:CPU:0*
_output_shapes
 �
Read_6/ReadVariableOpReadVariableOpYread_6_disablecopyonread_transformer_encoder_18_encoder_selfattentionlayer_1_query_kernel^Read_6/DisableCopyOnRead"/device:CPU:0*"
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
Read_7/DisableCopyOnReadDisableCopyOnReadWread_7_disablecopyonread_transformer_encoder_18_encoder_selfattentionlayer_1_query_bias"/device:CPU:0*
_output_shapes
 �
Read_7/ReadVariableOpReadVariableOpWread_7_disablecopyonread_transformer_encoder_18_encoder_selfattentionlayer_1_query_bias^Read_7/DisableCopyOnRead"/device:CPU:0*
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
Read_8/DisableCopyOnReadDisableCopyOnReadWread_8_disablecopyonread_transformer_encoder_18_encoder_selfattentionlayer_1_key_kernel"/device:CPU:0*
_output_shapes
 �
Read_8/ReadVariableOpReadVariableOpWread_8_disablecopyonread_transformer_encoder_18_encoder_selfattentionlayer_1_key_kernel^Read_8/DisableCopyOnRead"/device:CPU:0*"
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
Read_9/DisableCopyOnReadDisableCopyOnReadUread_9_disablecopyonread_transformer_encoder_18_encoder_selfattentionlayer_1_key_bias"/device:CPU:0*
_output_shapes
 �
Read_9/ReadVariableOpReadVariableOpUread_9_disablecopyonread_transformer_encoder_18_encoder_selfattentionlayer_1_key_bias^Read_9/DisableCopyOnRead"/device:CPU:0*
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
Read_10/DisableCopyOnReadDisableCopyOnReadZread_10_disablecopyonread_transformer_encoder_18_encoder_selfattentionlayer_1_value_kernel"/device:CPU:0*
_output_shapes
 �
Read_10/ReadVariableOpReadVariableOpZread_10_disablecopyonread_transformer_encoder_18_encoder_selfattentionlayer_1_value_kernel^Read_10/DisableCopyOnRead"/device:CPU:0*"
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
Read_11/DisableCopyOnReadDisableCopyOnReadXread_11_disablecopyonread_transformer_encoder_18_encoder_selfattentionlayer_1_value_bias"/device:CPU:0*
_output_shapes
 �
Read_11/ReadVariableOpReadVariableOpXread_11_disablecopyonread_transformer_encoder_18_encoder_selfattentionlayer_1_value_bias^Read_11/DisableCopyOnRead"/device:CPU:0*
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
Read_12/DisableCopyOnReadDisableCopyOnReaderead_12_disablecopyonread_transformer_encoder_18_encoder_selfattentionlayer_1_attention_output_kernel"/device:CPU:0*
_output_shapes
 �
Read_12/ReadVariableOpReadVariableOperead_12_disablecopyonread_transformer_encoder_18_encoder_selfattentionlayer_1_attention_output_kernel^Read_12/DisableCopyOnRead"/device:CPU:0*"
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
Read_13/DisableCopyOnReadDisableCopyOnReadcread_13_disablecopyonread_transformer_encoder_18_encoder_selfattentionlayer_1_attention_output_bias"/device:CPU:0*
_output_shapes
 �
Read_13/ReadVariableOpReadVariableOpcread_13_disablecopyonread_transformer_encoder_18_encoder_selfattentionlayer_1_attention_output_bias^Read_13/DisableCopyOnRead"/device:CPU:0*
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
Read_14/DisableCopyOnReadDisableCopyOnReadWread_14_disablecopyonread_transformer_encoder_18_encoder_1st_normalizationlayer_1_gamma"/device:CPU:0*
_output_shapes
 �
Read_14/ReadVariableOpReadVariableOpWread_14_disablecopyonread_transformer_encoder_18_encoder_1st_normalizationlayer_1_gamma^Read_14/DisableCopyOnRead"/device:CPU:0*
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
Read_15/DisableCopyOnReadDisableCopyOnReadVread_15_disablecopyonread_transformer_encoder_18_encoder_1st_normalizationlayer_1_beta"/device:CPU:0*
_output_shapes
 �
Read_15/ReadVariableOpReadVariableOpVread_15_disablecopyonread_transformer_encoder_18_encoder_1st_normalizationlayer_1_beta^Read_15/DisableCopyOnRead"/device:CPU:0*
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
Read_16/DisableCopyOnReadDisableCopyOnReadWread_16_disablecopyonread_transformer_encoder_18_encoder_2nd_normalizationlayer_1_gamma"/device:CPU:0*
_output_shapes
 �
Read_16/ReadVariableOpReadVariableOpWread_16_disablecopyonread_transformer_encoder_18_encoder_2nd_normalizationlayer_1_gamma^Read_16/DisableCopyOnRead"/device:CPU:0*
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
Read_17/DisableCopyOnReadDisableCopyOnReadVread_17_disablecopyonread_transformer_encoder_18_encoder_2nd_normalizationlayer_1_beta"/device:CPU:0*
_output_shapes
 �
Read_17/ReadVariableOpReadVariableOpVread_17_disablecopyonread_transformer_encoder_18_encoder_2nd_normalizationlayer_1_beta^Read_17/DisableCopyOnRead"/device:CPU:0*
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
Read_18/DisableCopyOnReadDisableCopyOnReadTread_18_disablecopyonread_transformer_encoder_18_encoder_feedforwardlayer_1_1_kernel"/device:CPU:0*
_output_shapes
 �
Read_18/ReadVariableOpReadVariableOpTread_18_disablecopyonread_transformer_encoder_18_encoder_feedforwardlayer_1_1_kernel^Read_18/DisableCopyOnRead"/device:CPU:0*
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
Read_19/DisableCopyOnReadDisableCopyOnReadRread_19_disablecopyonread_transformer_encoder_18_encoder_feedforwardlayer_1_1_bias"/device:CPU:0*
_output_shapes
 �
Read_19/ReadVariableOpReadVariableOpRread_19_disablecopyonread_transformer_encoder_18_encoder_feedforwardlayer_1_1_bias^Read_19/DisableCopyOnRead"/device:CPU:0*
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
Read_20/DisableCopyOnReadDisableCopyOnReadTread_20_disablecopyonread_transformer_encoder_18_encoder_feedforwardlayer_2_1_kernel"/device:CPU:0*
_output_shapes
 �
Read_20/ReadVariableOpReadVariableOpTread_20_disablecopyonread_transformer_encoder_18_encoder_feedforwardlayer_2_1_kernel^Read_20/DisableCopyOnRead"/device:CPU:0*
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
Read_21/DisableCopyOnReadDisableCopyOnReadRread_21_disablecopyonread_transformer_encoder_18_encoder_feedforwardlayer_2_1_bias"/device:CPU:0*
_output_shapes
 �
Read_21/ReadVariableOpReadVariableOpRread_21_disablecopyonread_transformer_encoder_18_encoder_feedforwardlayer_2_1_bias^Read_21/DisableCopyOnRead"/device:CPU:0*
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
Read_22/DisableCopyOnReadDisableCopyOnReadZread_22_disablecopyonread_transformer_encoder_19_encoder_selfattentionlayer_2_query_kernel"/device:CPU:0*
_output_shapes
 �
Read_22/ReadVariableOpReadVariableOpZread_22_disablecopyonread_transformer_encoder_19_encoder_selfattentionlayer_2_query_kernel^Read_22/DisableCopyOnRead"/device:CPU:0*"
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
Read_23/DisableCopyOnReadDisableCopyOnReadXread_23_disablecopyonread_transformer_encoder_19_encoder_selfattentionlayer_2_query_bias"/device:CPU:0*
_output_shapes
 �
Read_23/ReadVariableOpReadVariableOpXread_23_disablecopyonread_transformer_encoder_19_encoder_selfattentionlayer_2_query_bias^Read_23/DisableCopyOnRead"/device:CPU:0*
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
Read_24/DisableCopyOnReadDisableCopyOnReadXread_24_disablecopyonread_transformer_encoder_19_encoder_selfattentionlayer_2_key_kernel"/device:CPU:0*
_output_shapes
 �
Read_24/ReadVariableOpReadVariableOpXread_24_disablecopyonread_transformer_encoder_19_encoder_selfattentionlayer_2_key_kernel^Read_24/DisableCopyOnRead"/device:CPU:0*"
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
Read_25/DisableCopyOnReadDisableCopyOnReadVread_25_disablecopyonread_transformer_encoder_19_encoder_selfattentionlayer_2_key_bias"/device:CPU:0*
_output_shapes
 �
Read_25/ReadVariableOpReadVariableOpVread_25_disablecopyonread_transformer_encoder_19_encoder_selfattentionlayer_2_key_bias^Read_25/DisableCopyOnRead"/device:CPU:0*
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
Read_26/DisableCopyOnReadDisableCopyOnReadZread_26_disablecopyonread_transformer_encoder_19_encoder_selfattentionlayer_2_value_kernel"/device:CPU:0*
_output_shapes
 �
Read_26/ReadVariableOpReadVariableOpZread_26_disablecopyonread_transformer_encoder_19_encoder_selfattentionlayer_2_value_kernel^Read_26/DisableCopyOnRead"/device:CPU:0*"
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
Read_27/DisableCopyOnReadDisableCopyOnReadXread_27_disablecopyonread_transformer_encoder_19_encoder_selfattentionlayer_2_value_bias"/device:CPU:0*
_output_shapes
 �
Read_27/ReadVariableOpReadVariableOpXread_27_disablecopyonread_transformer_encoder_19_encoder_selfattentionlayer_2_value_bias^Read_27/DisableCopyOnRead"/device:CPU:0*
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
Read_28/DisableCopyOnReadDisableCopyOnReaderead_28_disablecopyonread_transformer_encoder_19_encoder_selfattentionlayer_2_attention_output_kernel"/device:CPU:0*
_output_shapes
 �
Read_28/ReadVariableOpReadVariableOperead_28_disablecopyonread_transformer_encoder_19_encoder_selfattentionlayer_2_attention_output_kernel^Read_28/DisableCopyOnRead"/device:CPU:0*"
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
Read_29/DisableCopyOnReadDisableCopyOnReadcread_29_disablecopyonread_transformer_encoder_19_encoder_selfattentionlayer_2_attention_output_bias"/device:CPU:0*
_output_shapes
 �
Read_29/ReadVariableOpReadVariableOpcread_29_disablecopyonread_transformer_encoder_19_encoder_selfattentionlayer_2_attention_output_bias^Read_29/DisableCopyOnRead"/device:CPU:0*
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
Read_30/DisableCopyOnReadDisableCopyOnReadWread_30_disablecopyonread_transformer_encoder_19_encoder_1st_normalizationlayer_2_gamma"/device:CPU:0*
_output_shapes
 �
Read_30/ReadVariableOpReadVariableOpWread_30_disablecopyonread_transformer_encoder_19_encoder_1st_normalizationlayer_2_gamma^Read_30/DisableCopyOnRead"/device:CPU:0*
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
Read_31/DisableCopyOnReadDisableCopyOnReadVread_31_disablecopyonread_transformer_encoder_19_encoder_1st_normalizationlayer_2_beta"/device:CPU:0*
_output_shapes
 �
Read_31/ReadVariableOpReadVariableOpVread_31_disablecopyonread_transformer_encoder_19_encoder_1st_normalizationlayer_2_beta^Read_31/DisableCopyOnRead"/device:CPU:0*
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
Read_32/DisableCopyOnReadDisableCopyOnReadWread_32_disablecopyonread_transformer_encoder_19_encoder_2nd_normalizationlayer_2_gamma"/device:CPU:0*
_output_shapes
 �
Read_32/ReadVariableOpReadVariableOpWread_32_disablecopyonread_transformer_encoder_19_encoder_2nd_normalizationlayer_2_gamma^Read_32/DisableCopyOnRead"/device:CPU:0*
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
Read_33/DisableCopyOnReadDisableCopyOnReadVread_33_disablecopyonread_transformer_encoder_19_encoder_2nd_normalizationlayer_2_beta"/device:CPU:0*
_output_shapes
 �
Read_33/ReadVariableOpReadVariableOpVread_33_disablecopyonread_transformer_encoder_19_encoder_2nd_normalizationlayer_2_beta^Read_33/DisableCopyOnRead"/device:CPU:0*
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
Read_34/DisableCopyOnReadDisableCopyOnReadTread_34_disablecopyonread_transformer_encoder_19_encoder_feedforwardlayer_1_2_kernel"/device:CPU:0*
_output_shapes
 �
Read_34/ReadVariableOpReadVariableOpTread_34_disablecopyonread_transformer_encoder_19_encoder_feedforwardlayer_1_2_kernel^Read_34/DisableCopyOnRead"/device:CPU:0*
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
Read_35/DisableCopyOnReadDisableCopyOnReadRread_35_disablecopyonread_transformer_encoder_19_encoder_feedforwardlayer_1_2_bias"/device:CPU:0*
_output_shapes
 �
Read_35/ReadVariableOpReadVariableOpRread_35_disablecopyonread_transformer_encoder_19_encoder_feedforwardlayer_1_2_bias^Read_35/DisableCopyOnRead"/device:CPU:0*
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
Read_36/DisableCopyOnReadDisableCopyOnReadTread_36_disablecopyonread_transformer_encoder_19_encoder_feedforwardlayer_2_2_kernel"/device:CPU:0*
_output_shapes
 �
Read_36/ReadVariableOpReadVariableOpTread_36_disablecopyonread_transformer_encoder_19_encoder_feedforwardlayer_2_2_kernel^Read_36/DisableCopyOnRead"/device:CPU:0*
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
Read_37/DisableCopyOnReadDisableCopyOnReadRread_37_disablecopyonread_transformer_encoder_19_encoder_feedforwardlayer_2_2_bias"/device:CPU:0*
_output_shapes
 �
Read_37/ReadVariableOpReadVariableOpRread_37_disablecopyonread_transformer_encoder_19_encoder_feedforwardlayer_2_2_bias^Read_37/DisableCopyOnRead"/device:CPU:0*
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
Read_38/DisableCopyOnReadDisableCopyOnReadZread_38_disablecopyonread_transformer_encoder_20_encoder_selfattentionlayer_3_query_kernel"/device:CPU:0*
_output_shapes
 �
Read_38/ReadVariableOpReadVariableOpZread_38_disablecopyonread_transformer_encoder_20_encoder_selfattentionlayer_3_query_kernel^Read_38/DisableCopyOnRead"/device:CPU:0*"
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
Read_39/DisableCopyOnReadDisableCopyOnReadXread_39_disablecopyonread_transformer_encoder_20_encoder_selfattentionlayer_3_query_bias"/device:CPU:0*
_output_shapes
 �
Read_39/ReadVariableOpReadVariableOpXread_39_disablecopyonread_transformer_encoder_20_encoder_selfattentionlayer_3_query_bias^Read_39/DisableCopyOnRead"/device:CPU:0*
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
Read_40/DisableCopyOnReadDisableCopyOnReadXread_40_disablecopyonread_transformer_encoder_20_encoder_selfattentionlayer_3_key_kernel"/device:CPU:0*
_output_shapes
 �
Read_40/ReadVariableOpReadVariableOpXread_40_disablecopyonread_transformer_encoder_20_encoder_selfattentionlayer_3_key_kernel^Read_40/DisableCopyOnRead"/device:CPU:0*"
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
Read_41/DisableCopyOnReadDisableCopyOnReadVread_41_disablecopyonread_transformer_encoder_20_encoder_selfattentionlayer_3_key_bias"/device:CPU:0*
_output_shapes
 �
Read_41/ReadVariableOpReadVariableOpVread_41_disablecopyonread_transformer_encoder_20_encoder_selfattentionlayer_3_key_bias^Read_41/DisableCopyOnRead"/device:CPU:0*
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
Read_42/DisableCopyOnReadDisableCopyOnReadZread_42_disablecopyonread_transformer_encoder_20_encoder_selfattentionlayer_3_value_kernel"/device:CPU:0*
_output_shapes
 �
Read_42/ReadVariableOpReadVariableOpZread_42_disablecopyonread_transformer_encoder_20_encoder_selfattentionlayer_3_value_kernel^Read_42/DisableCopyOnRead"/device:CPU:0*"
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
Read_43/DisableCopyOnReadDisableCopyOnReadXread_43_disablecopyonread_transformer_encoder_20_encoder_selfattentionlayer_3_value_bias"/device:CPU:0*
_output_shapes
 �
Read_43/ReadVariableOpReadVariableOpXread_43_disablecopyonread_transformer_encoder_20_encoder_selfattentionlayer_3_value_bias^Read_43/DisableCopyOnRead"/device:CPU:0*
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
Read_44/DisableCopyOnReadDisableCopyOnReaderead_44_disablecopyonread_transformer_encoder_20_encoder_selfattentionlayer_3_attention_output_kernel"/device:CPU:0*
_output_shapes
 �
Read_44/ReadVariableOpReadVariableOperead_44_disablecopyonread_transformer_encoder_20_encoder_selfattentionlayer_3_attention_output_kernel^Read_44/DisableCopyOnRead"/device:CPU:0*"
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
Read_45/DisableCopyOnReadDisableCopyOnReadcread_45_disablecopyonread_transformer_encoder_20_encoder_selfattentionlayer_3_attention_output_bias"/device:CPU:0*
_output_shapes
 �
Read_45/ReadVariableOpReadVariableOpcread_45_disablecopyonread_transformer_encoder_20_encoder_selfattentionlayer_3_attention_output_bias^Read_45/DisableCopyOnRead"/device:CPU:0*
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
Read_46/DisableCopyOnReadDisableCopyOnReadWread_46_disablecopyonread_transformer_encoder_20_encoder_1st_normalizationlayer_3_gamma"/device:CPU:0*
_output_shapes
 �
Read_46/ReadVariableOpReadVariableOpWread_46_disablecopyonread_transformer_encoder_20_encoder_1st_normalizationlayer_3_gamma^Read_46/DisableCopyOnRead"/device:CPU:0*
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
Read_47/DisableCopyOnReadDisableCopyOnReadVread_47_disablecopyonread_transformer_encoder_20_encoder_1st_normalizationlayer_3_beta"/device:CPU:0*
_output_shapes
 �
Read_47/ReadVariableOpReadVariableOpVread_47_disablecopyonread_transformer_encoder_20_encoder_1st_normalizationlayer_3_beta^Read_47/DisableCopyOnRead"/device:CPU:0*
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
Read_48/DisableCopyOnReadDisableCopyOnReadWread_48_disablecopyonread_transformer_encoder_20_encoder_2nd_normalizationlayer_3_gamma"/device:CPU:0*
_output_shapes
 �
Read_48/ReadVariableOpReadVariableOpWread_48_disablecopyonread_transformer_encoder_20_encoder_2nd_normalizationlayer_3_gamma^Read_48/DisableCopyOnRead"/device:CPU:0*
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
Read_49/DisableCopyOnReadDisableCopyOnReadVread_49_disablecopyonread_transformer_encoder_20_encoder_2nd_normalizationlayer_3_beta"/device:CPU:0*
_output_shapes
 �
Read_49/ReadVariableOpReadVariableOpVread_49_disablecopyonread_transformer_encoder_20_encoder_2nd_normalizationlayer_3_beta^Read_49/DisableCopyOnRead"/device:CPU:0*
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
Read_50/DisableCopyOnReadDisableCopyOnReadTread_50_disablecopyonread_transformer_encoder_20_encoder_feedforwardlayer_1_3_kernel"/device:CPU:0*
_output_shapes
 �
Read_50/ReadVariableOpReadVariableOpTread_50_disablecopyonread_transformer_encoder_20_encoder_feedforwardlayer_1_3_kernel^Read_50/DisableCopyOnRead"/device:CPU:0*
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
Read_51/DisableCopyOnReadDisableCopyOnReadRread_51_disablecopyonread_transformer_encoder_20_encoder_feedforwardlayer_1_3_bias"/device:CPU:0*
_output_shapes
 �
Read_51/ReadVariableOpReadVariableOpRread_51_disablecopyonread_transformer_encoder_20_encoder_feedforwardlayer_1_3_bias^Read_51/DisableCopyOnRead"/device:CPU:0*
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
Read_52/DisableCopyOnReadDisableCopyOnReadTread_52_disablecopyonread_transformer_encoder_20_encoder_feedforwardlayer_2_3_kernel"/device:CPU:0*
_output_shapes
 �
Read_52/ReadVariableOpReadVariableOpTread_52_disablecopyonread_transformer_encoder_20_encoder_feedforwardlayer_2_3_kernel^Read_52/DisableCopyOnRead"/device:CPU:0*
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
Read_53/DisableCopyOnReadDisableCopyOnReadRread_53_disablecopyonread_transformer_encoder_20_encoder_feedforwardlayer_2_3_bias"/device:CPU:0*
_output_shapes
 �
Read_53/ReadVariableOpReadVariableOpRread_53_disablecopyonread_transformer_encoder_20_encoder_feedforwardlayer_2_3_bias^Read_53/DisableCopyOnRead"/device:CPU:0*
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

_user_specified_nameConst:X6T
R
_user_specified_name:8transformer_encoder_20/Encoder-FeedForwardLayer_2_3/bias:Z5V
T
_user_specified_name<:transformer_encoder_20/Encoder-FeedForwardLayer_2_3/kernel:X4T
R
_user_specified_name:8transformer_encoder_20/Encoder-FeedForwardLayer_1_3/bias:Z3V
T
_user_specified_name<:transformer_encoder_20/Encoder-FeedForwardLayer_1_3/kernel:\2X
V
_user_specified_name><transformer_encoder_20/Encoder-2nd-NormalizationLayer-3/beta:]1Y
W
_user_specified_name?=transformer_encoder_20/Encoder-2nd-NormalizationLayer-3/gamma:\0X
V
_user_specified_name><transformer_encoder_20/Encoder-1st-NormalizationLayer-3/beta:]/Y
W
_user_specified_name?=transformer_encoder_20/Encoder-1st-NormalizationLayer-3/gamma:i.e
c
_user_specified_nameKItransformer_encoder_20/Encoder-SelfAttentionLayer-3/attention_output/bias:k-g
e
_user_specified_nameMKtransformer_encoder_20/Encoder-SelfAttentionLayer-3/attention_output/kernel:^,Z
X
_user_specified_name@>transformer_encoder_20/Encoder-SelfAttentionLayer-3/value/bias:`+\
Z
_user_specified_nameB@transformer_encoder_20/Encoder-SelfAttentionLayer-3/value/kernel:\*X
V
_user_specified_name><transformer_encoder_20/Encoder-SelfAttentionLayer-3/key/bias:^)Z
X
_user_specified_name@>transformer_encoder_20/Encoder-SelfAttentionLayer-3/key/kernel:^(Z
X
_user_specified_name@>transformer_encoder_20/Encoder-SelfAttentionLayer-3/query/bias:`'\
Z
_user_specified_nameB@transformer_encoder_20/Encoder-SelfAttentionLayer-3/query/kernel:X&T
R
_user_specified_name:8transformer_encoder_19/Encoder-FeedForwardLayer_2_2/bias:Z%V
T
_user_specified_name<:transformer_encoder_19/Encoder-FeedForwardLayer_2_2/kernel:X$T
R
_user_specified_name:8transformer_encoder_19/Encoder-FeedForwardLayer_1_2/bias:Z#V
T
_user_specified_name<:transformer_encoder_19/Encoder-FeedForwardLayer_1_2/kernel:\"X
V
_user_specified_name><transformer_encoder_19/Encoder-2nd-NormalizationLayer-2/beta:]!Y
W
_user_specified_name?=transformer_encoder_19/Encoder-2nd-NormalizationLayer-2/gamma:\ X
V
_user_specified_name><transformer_encoder_19/Encoder-1st-NormalizationLayer-2/beta:]Y
W
_user_specified_name?=transformer_encoder_19/Encoder-1st-NormalizationLayer-2/gamma:ie
c
_user_specified_nameKItransformer_encoder_19/Encoder-SelfAttentionLayer-2/attention_output/bias:kg
e
_user_specified_nameMKtransformer_encoder_19/Encoder-SelfAttentionLayer-2/attention_output/kernel:^Z
X
_user_specified_name@>transformer_encoder_19/Encoder-SelfAttentionLayer-2/value/bias:`\
Z
_user_specified_nameB@transformer_encoder_19/Encoder-SelfAttentionLayer-2/value/kernel:\X
V
_user_specified_name><transformer_encoder_19/Encoder-SelfAttentionLayer-2/key/bias:^Z
X
_user_specified_name@>transformer_encoder_19/Encoder-SelfAttentionLayer-2/key/kernel:^Z
X
_user_specified_name@>transformer_encoder_19/Encoder-SelfAttentionLayer-2/query/bias:`\
Z
_user_specified_nameB@transformer_encoder_19/Encoder-SelfAttentionLayer-2/query/kernel:XT
R
_user_specified_name:8transformer_encoder_18/Encoder-FeedForwardLayer_2_1/bias:ZV
T
_user_specified_name<:transformer_encoder_18/Encoder-FeedForwardLayer_2_1/kernel:XT
R
_user_specified_name:8transformer_encoder_18/Encoder-FeedForwardLayer_1_1/bias:ZV
T
_user_specified_name<:transformer_encoder_18/Encoder-FeedForwardLayer_1_1/kernel:\X
V
_user_specified_name><transformer_encoder_18/Encoder-2nd-NormalizationLayer-1/beta:]Y
W
_user_specified_name?=transformer_encoder_18/Encoder-2nd-NormalizationLayer-1/gamma:\X
V
_user_specified_name><transformer_encoder_18/Encoder-1st-NormalizationLayer-1/beta:]Y
W
_user_specified_name?=transformer_encoder_18/Encoder-1st-NormalizationLayer-1/gamma:ie
c
_user_specified_nameKItransformer_encoder_18/Encoder-SelfAttentionLayer-1/attention_output/bias:kg
e
_user_specified_nameMKtransformer_encoder_18/Encoder-SelfAttentionLayer-1/attention_output/kernel:^Z
X
_user_specified_name@>transformer_encoder_18/Encoder-SelfAttentionLayer-1/value/bias:`\
Z
_user_specified_nameB@transformer_encoder_18/Encoder-SelfAttentionLayer-1/value/kernel:\
X
V
_user_specified_name><transformer_encoder_18/Encoder-SelfAttentionLayer-1/key/bias:^	Z
X
_user_specified_name@>transformer_encoder_18/Encoder-SelfAttentionLayer-1/key/kernel:^Z
X
_user_specified_name@>transformer_encoder_18/Encoder-SelfAttentionLayer-1/query/bias:`\
Z
_user_specified_nameB@transformer_encoder_18/Encoder-SelfAttentionLayer-1/query/kernel:84
2
_user_specified_namePredictionAreaRatio/bias::6
4
_user_specified_namePredictionAreaRatio/kernel:A=
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
�

�
P__inference_PredictionAreaRatio_layer_call_and_return_conditional_losses_1472973

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
m
Q__inference_StandardizeTimeLimit_layer_call_and_return_conditional_losses_1472930

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
�,
�
)__inference_model_6_layer_call_fn_1473772
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
GPU2*0J 8� *M
fHRF
D__inference_model_6_layer_call_and_return_conditional_losses_1472986`
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
StatefulPartitionedCallStatefulPartitionedCall:'7#
!
_user_specified_name	1473768:'6#
!
_user_specified_name	1473766:'5#
!
_user_specified_name	1473764:'4#
!
_user_specified_name	1473762:'3#
!
_user_specified_name	1473760:'2#
!
_user_specified_name	1473758:'1#
!
_user_specified_name	1473756:'0#
!
_user_specified_name	1473754:'/#
!
_user_specified_name	1473752:'.#
!
_user_specified_name	1473750:'-#
!
_user_specified_name	1473748:',#
!
_user_specified_name	1473746:'+#
!
_user_specified_name	1473744:'*#
!
_user_specified_name	1473742:')#
!
_user_specified_name	1473740:'(#
!
_user_specified_name	1473738:''#
!
_user_specified_name	1473736:'&#
!
_user_specified_name	1473734:'%#
!
_user_specified_name	1473732:'$#
!
_user_specified_name	1473730:'##
!
_user_specified_name	1473728:'"#
!
_user_specified_name	1473726:'!#
!
_user_specified_name	1473724:' #
!
_user_specified_name	1473722:'#
!
_user_specified_name	1473720:'#
!
_user_specified_name	1473718:'#
!
_user_specified_name	1473716:'#
!
_user_specified_name	1473714:'#
!
_user_specified_name	1473712:'#
!
_user_specified_name	1473710:'#
!
_user_specified_name	1473708:'#
!
_user_specified_name	1473706:'#
!
_user_specified_name	1473704:'#
!
_user_specified_name	1473702:'#
!
_user_specified_name	1473700:'#
!
_user_specified_name	1473698:'#
!
_user_specified_name	1473696:'#
!
_user_specified_name	1473694:'#
!
_user_specified_name	1473692:'#
!
_user_specified_name	1473690:'#
!
_user_specified_name	1473688:'#
!
_user_specified_name	1473686:'#
!
_user_specified_name	1473684:'#
!
_user_specified_name	1473682:'#
!
_user_specified_name	1473680:'
#
!
_user_specified_name	1473678:'	#
!
_user_specified_name	1473676:'#
!
_user_specified_name	1473674:'#
!
_user_specified_name	1473672:'#
!
_user_specified_name	1473670:'#
!
_user_specified_name	1473668:'#
!
_user_specified_name	1473666:'#
!
_user_specified_name	1473664:'#
!
_user_specified_name	1473662:WS
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
�S
"__inference__wrapped_model_1472183
stacklevelinputfeatures
timelimitinputi
[model_6_transformer_encoder_18_encoder_1st_normalizationlayer_1_mul_readvariableop_resource:	i
[model_6_transformer_encoder_18_encoder_1st_normalizationlayer_1_add_readvariableop_resource:	}
gmodel_6_transformer_encoder_18_encoder_selfattentionlayer_1_query_einsum_einsum_readvariableop_resource:	o
]model_6_transformer_encoder_18_encoder_selfattentionlayer_1_query_add_readvariableop_resource:{
emodel_6_transformer_encoder_18_encoder_selfattentionlayer_1_key_einsum_einsum_readvariableop_resource:	m
[model_6_transformer_encoder_18_encoder_selfattentionlayer_1_key_add_readvariableop_resource:}
gmodel_6_transformer_encoder_18_encoder_selfattentionlayer_1_value_einsum_einsum_readvariableop_resource:	o
]model_6_transformer_encoder_18_encoder_selfattentionlayer_1_value_add_readvariableop_resource:�
rmodel_6_transformer_encoder_18_encoder_selfattentionlayer_1_attention_output_einsum_einsum_readvariableop_resource:	v
hmodel_6_transformer_encoder_18_encoder_selfattentionlayer_1_attention_output_add_readvariableop_resource:	i
[model_6_transformer_encoder_18_encoder_2nd_normalizationlayer_1_mul_readvariableop_resource:	i
[model_6_transformer_encoder_18_encoder_2nd_normalizationlayer_1_add_readvariableop_resource:	o
]model_6_transformer_encoder_18_encoder_feedforwardlayer_1_1_tensordot_readvariableop_resource:		i
[model_6_transformer_encoder_18_encoder_feedforwardlayer_1_1_biasadd_readvariableop_resource:	o
]model_6_transformer_encoder_18_encoder_feedforwardlayer_2_1_tensordot_readvariableop_resource:		i
[model_6_transformer_encoder_18_encoder_feedforwardlayer_2_1_biasadd_readvariableop_resource:	i
[model_6_transformer_encoder_19_encoder_1st_normalizationlayer_2_mul_readvariableop_resource:	i
[model_6_transformer_encoder_19_encoder_1st_normalizationlayer_2_add_readvariableop_resource:	}
gmodel_6_transformer_encoder_19_encoder_selfattentionlayer_2_query_einsum_einsum_readvariableop_resource:	o
]model_6_transformer_encoder_19_encoder_selfattentionlayer_2_query_add_readvariableop_resource:{
emodel_6_transformer_encoder_19_encoder_selfattentionlayer_2_key_einsum_einsum_readvariableop_resource:	m
[model_6_transformer_encoder_19_encoder_selfattentionlayer_2_key_add_readvariableop_resource:}
gmodel_6_transformer_encoder_19_encoder_selfattentionlayer_2_value_einsum_einsum_readvariableop_resource:	o
]model_6_transformer_encoder_19_encoder_selfattentionlayer_2_value_add_readvariableop_resource:�
rmodel_6_transformer_encoder_19_encoder_selfattentionlayer_2_attention_output_einsum_einsum_readvariableop_resource:	v
hmodel_6_transformer_encoder_19_encoder_selfattentionlayer_2_attention_output_add_readvariableop_resource:	i
[model_6_transformer_encoder_19_encoder_2nd_normalizationlayer_2_mul_readvariableop_resource:	i
[model_6_transformer_encoder_19_encoder_2nd_normalizationlayer_2_add_readvariableop_resource:	o
]model_6_transformer_encoder_19_encoder_feedforwardlayer_1_2_tensordot_readvariableop_resource:		i
[model_6_transformer_encoder_19_encoder_feedforwardlayer_1_2_biasadd_readvariableop_resource:	o
]model_6_transformer_encoder_19_encoder_feedforwardlayer_2_2_tensordot_readvariableop_resource:		i
[model_6_transformer_encoder_19_encoder_feedforwardlayer_2_2_biasadd_readvariableop_resource:	i
[model_6_transformer_encoder_20_encoder_1st_normalizationlayer_3_mul_readvariableop_resource:	i
[model_6_transformer_encoder_20_encoder_1st_normalizationlayer_3_add_readvariableop_resource:	}
gmodel_6_transformer_encoder_20_encoder_selfattentionlayer_3_query_einsum_einsum_readvariableop_resource:	o
]model_6_transformer_encoder_20_encoder_selfattentionlayer_3_query_add_readvariableop_resource:{
emodel_6_transformer_encoder_20_encoder_selfattentionlayer_3_key_einsum_einsum_readvariableop_resource:	m
[model_6_transformer_encoder_20_encoder_selfattentionlayer_3_key_add_readvariableop_resource:}
gmodel_6_transformer_encoder_20_encoder_selfattentionlayer_3_value_einsum_einsum_readvariableop_resource:	o
]model_6_transformer_encoder_20_encoder_selfattentionlayer_3_value_add_readvariableop_resource:�
rmodel_6_transformer_encoder_20_encoder_selfattentionlayer_3_attention_output_einsum_einsum_readvariableop_resource:	v
hmodel_6_transformer_encoder_20_encoder_selfattentionlayer_3_attention_output_add_readvariableop_resource:	i
[model_6_transformer_encoder_20_encoder_2nd_normalizationlayer_3_mul_readvariableop_resource:	i
[model_6_transformer_encoder_20_encoder_2nd_normalizationlayer_3_add_readvariableop_resource:	o
]model_6_transformer_encoder_20_encoder_feedforwardlayer_1_3_tensordot_readvariableop_resource:		i
[model_6_transformer_encoder_20_encoder_feedforwardlayer_1_3_biasadd_readvariableop_resource:	o
]model_6_transformer_encoder_20_encoder_feedforwardlayer_2_3_tensordot_readvariableop_resource:		i
[model_6_transformer_encoder_20_encoder_feedforwardlayer_2_3_biasadd_readvariableop_resource:	@
2model_6_finallayernorm_mul_readvariableop_resource:	@
2model_6_finallayernorm_add_readvariableop_resource:	U
Cmodel_6_fullyconnectedlayerarearatio_matmul_readvariableop_resource:

R
Dmodel_6_fullyconnectedlayerarearatio_biasadd_readvariableop_resource:
L
:model_6_predictionarearatio_matmul_readvariableop_resource:
I
;model_6_predictionarearatio_biasadd_readvariableop_resource:
identity��)model_6/FinalLayerNorm/add/ReadVariableOp�)model_6/FinalLayerNorm/mul/ReadVariableOp�;model_6/FullyConnectedLayerAreaRatio/BiasAdd/ReadVariableOp�:model_6/FullyConnectedLayerAreaRatio/MatMul/ReadVariableOp�2model_6/PredictionAreaRatio/BiasAdd/ReadVariableOp�1model_6/PredictionAreaRatio/MatMul/ReadVariableOp�Rmodel_6/transformer_encoder_18/Encoder-1st-NormalizationLayer-1/add/ReadVariableOp�Rmodel_6/transformer_encoder_18/Encoder-1st-NormalizationLayer-1/mul/ReadVariableOp�Rmodel_6/transformer_encoder_18/Encoder-2nd-NormalizationLayer-1/add/ReadVariableOp�Rmodel_6/transformer_encoder_18/Encoder-2nd-NormalizationLayer-1/mul/ReadVariableOp�Rmodel_6/transformer_encoder_18/Encoder-FeedForwardLayer_1_1/BiasAdd/ReadVariableOp�Tmodel_6/transformer_encoder_18/Encoder-FeedForwardLayer_1_1/Tensordot/ReadVariableOp�Rmodel_6/transformer_encoder_18/Encoder-FeedForwardLayer_2_1/BiasAdd/ReadVariableOp�Tmodel_6/transformer_encoder_18/Encoder-FeedForwardLayer_2_1/Tensordot/ReadVariableOp�_model_6/transformer_encoder_18/Encoder-SelfAttentionLayer-1/attention_output/add/ReadVariableOp�imodel_6/transformer_encoder_18/Encoder-SelfAttentionLayer-1/attention_output/einsum/Einsum/ReadVariableOp�Rmodel_6/transformer_encoder_18/Encoder-SelfAttentionLayer-1/key/add/ReadVariableOp�\model_6/transformer_encoder_18/Encoder-SelfAttentionLayer-1/key/einsum/Einsum/ReadVariableOp�Tmodel_6/transformer_encoder_18/Encoder-SelfAttentionLayer-1/query/add/ReadVariableOp�^model_6/transformer_encoder_18/Encoder-SelfAttentionLayer-1/query/einsum/Einsum/ReadVariableOp�Tmodel_6/transformer_encoder_18/Encoder-SelfAttentionLayer-1/value/add/ReadVariableOp�^model_6/transformer_encoder_18/Encoder-SelfAttentionLayer-1/value/einsum/Einsum/ReadVariableOp�Rmodel_6/transformer_encoder_19/Encoder-1st-NormalizationLayer-2/add/ReadVariableOp�Rmodel_6/transformer_encoder_19/Encoder-1st-NormalizationLayer-2/mul/ReadVariableOp�Rmodel_6/transformer_encoder_19/Encoder-2nd-NormalizationLayer-2/add/ReadVariableOp�Rmodel_6/transformer_encoder_19/Encoder-2nd-NormalizationLayer-2/mul/ReadVariableOp�Rmodel_6/transformer_encoder_19/Encoder-FeedForwardLayer_1_2/BiasAdd/ReadVariableOp�Tmodel_6/transformer_encoder_19/Encoder-FeedForwardLayer_1_2/Tensordot/ReadVariableOp�Rmodel_6/transformer_encoder_19/Encoder-FeedForwardLayer_2_2/BiasAdd/ReadVariableOp�Tmodel_6/transformer_encoder_19/Encoder-FeedForwardLayer_2_2/Tensordot/ReadVariableOp�_model_6/transformer_encoder_19/Encoder-SelfAttentionLayer-2/attention_output/add/ReadVariableOp�imodel_6/transformer_encoder_19/Encoder-SelfAttentionLayer-2/attention_output/einsum/Einsum/ReadVariableOp�Rmodel_6/transformer_encoder_19/Encoder-SelfAttentionLayer-2/key/add/ReadVariableOp�\model_6/transformer_encoder_19/Encoder-SelfAttentionLayer-2/key/einsum/Einsum/ReadVariableOp�Tmodel_6/transformer_encoder_19/Encoder-SelfAttentionLayer-2/query/add/ReadVariableOp�^model_6/transformer_encoder_19/Encoder-SelfAttentionLayer-2/query/einsum/Einsum/ReadVariableOp�Tmodel_6/transformer_encoder_19/Encoder-SelfAttentionLayer-2/value/add/ReadVariableOp�^model_6/transformer_encoder_19/Encoder-SelfAttentionLayer-2/value/einsum/Einsum/ReadVariableOp�Rmodel_6/transformer_encoder_20/Encoder-1st-NormalizationLayer-3/add/ReadVariableOp�Rmodel_6/transformer_encoder_20/Encoder-1st-NormalizationLayer-3/mul/ReadVariableOp�Rmodel_6/transformer_encoder_20/Encoder-2nd-NormalizationLayer-3/add/ReadVariableOp�Rmodel_6/transformer_encoder_20/Encoder-2nd-NormalizationLayer-3/mul/ReadVariableOp�Rmodel_6/transformer_encoder_20/Encoder-FeedForwardLayer_1_3/BiasAdd/ReadVariableOp�Tmodel_6/transformer_encoder_20/Encoder-FeedForwardLayer_1_3/Tensordot/ReadVariableOp�Rmodel_6/transformer_encoder_20/Encoder-FeedForwardLayer_2_3/BiasAdd/ReadVariableOp�Tmodel_6/transformer_encoder_20/Encoder-FeedForwardLayer_2_3/Tensordot/ReadVariableOp�_model_6/transformer_encoder_20/Encoder-SelfAttentionLayer-3/attention_output/add/ReadVariableOp�imodel_6/transformer_encoder_20/Encoder-SelfAttentionLayer-3/attention_output/einsum/Einsum/ReadVariableOp�Rmodel_6/transformer_encoder_20/Encoder-SelfAttentionLayer-3/key/add/ReadVariableOp�\model_6/transformer_encoder_20/Encoder-SelfAttentionLayer-3/key/einsum/Einsum/ReadVariableOp�Tmodel_6/transformer_encoder_20/Encoder-SelfAttentionLayer-3/query/add/ReadVariableOp�^model_6/transformer_encoder_20/Encoder-SelfAttentionLayer-3/query/einsum/Einsum/ReadVariableOp�Tmodel_6/transformer_encoder_20/Encoder-SelfAttentionLayer-3/value/add/ReadVariableOp�^model_6/transformer_encoder_20/Encoder-SelfAttentionLayer-3/value/einsum/Einsum/ReadVariableOpa
model_6/MaskingLayer/NotEqual/yConst*
_output_shapes
: *
dtype0*
value	B j �
model_6/MaskingLayer/NotEqualNotEqualstacklevelinputfeatures(model_6/MaskingLayer/NotEqual/y:output:0*
T0*+
_output_shapes
:���������P	u
*model_6/MaskingLayer/Any/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
����������
model_6/MaskingLayer/AnyAny!model_6/MaskingLayer/NotEqual:z:03model_6/MaskingLayer/Any/reduction_indices:output:0*+
_output_shapes
:���������P*
	keep_dims(�
model_6/MaskingLayer/CastCast!model_6/MaskingLayer/Any:output:0*

DstT0*

SrcT0
*+
_output_shapes
:���������P�
model_6/MaskingLayer/mulMulstacklevelinputfeaturesmodel_6/MaskingLayer/Cast:y:0*
T0*+
_output_shapes
:���������P	�
model_6/MaskingLayer/SqueezeSqueeze!model_6/MaskingLayer/Any:output:0*
T0
*'
_output_shapes
:���������P*
squeeze_dims

����������
#model_6/transformer_encoder_18/CastCastmodel_6/MaskingLayer/mul:z:0*

DstT0*

SrcT0*+
_output_shapes
:���������P	�
Emodel_6/transformer_encoder_18/Encoder-1st-NormalizationLayer-1/ShapeShape'model_6/transformer_encoder_18/Cast:y:0*
T0*
_output_shapes
::���
Smodel_6/transformer_encoder_18/Encoder-1st-NormalizationLayer-1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
Umodel_6/transformer_encoder_18/Encoder-1st-NormalizationLayer-1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
Umodel_6/transformer_encoder_18/Encoder-1st-NormalizationLayer-1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
Mmodel_6/transformer_encoder_18/Encoder-1st-NormalizationLayer-1/strided_sliceStridedSliceNmodel_6/transformer_encoder_18/Encoder-1st-NormalizationLayer-1/Shape:output:0\model_6/transformer_encoder_18/Encoder-1st-NormalizationLayer-1/strided_slice/stack:output:0^model_6/transformer_encoder_18/Encoder-1st-NormalizationLayer-1/strided_slice/stack_1:output:0^model_6/transformer_encoder_18/Encoder-1st-NormalizationLayer-1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask�
Emodel_6/transformer_encoder_18/Encoder-1st-NormalizationLayer-1/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
Dmodel_6/transformer_encoder_18/Encoder-1st-NormalizationLayer-1/ProdProdVmodel_6/transformer_encoder_18/Encoder-1st-NormalizationLayer-1/strided_slice:output:0Nmodel_6/transformer_encoder_18/Encoder-1st-NormalizationLayer-1/Const:output:0*
T0*
_output_shapes
: �
Umodel_6/transformer_encoder_18/Encoder-1st-NormalizationLayer-1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:�
Wmodel_6/transformer_encoder_18/Encoder-1st-NormalizationLayer-1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: �
Wmodel_6/transformer_encoder_18/Encoder-1st-NormalizationLayer-1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
Omodel_6/transformer_encoder_18/Encoder-1st-NormalizationLayer-1/strided_slice_1StridedSliceNmodel_6/transformer_encoder_18/Encoder-1st-NormalizationLayer-1/Shape:output:0^model_6/transformer_encoder_18/Encoder-1st-NormalizationLayer-1/strided_slice_1/stack:output:0`model_6/transformer_encoder_18/Encoder-1st-NormalizationLayer-1/strided_slice_1/stack_1:output:0`model_6/transformer_encoder_18/Encoder-1st-NormalizationLayer-1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask�
Gmodel_6/transformer_encoder_18/Encoder-1st-NormalizationLayer-1/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
Fmodel_6/transformer_encoder_18/Encoder-1st-NormalizationLayer-1/Prod_1ProdXmodel_6/transformer_encoder_18/Encoder-1st-NormalizationLayer-1/strided_slice_1:output:0Pmodel_6/transformer_encoder_18/Encoder-1st-NormalizationLayer-1/Const_1:output:0*
T0*
_output_shapes
: �
Omodel_6/transformer_encoder_18/Encoder-1st-NormalizationLayer-1/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :�
Omodel_6/transformer_encoder_18/Encoder-1st-NormalizationLayer-1/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :�
Mmodel_6/transformer_encoder_18/Encoder-1st-NormalizationLayer-1/Reshape/shapePackXmodel_6/transformer_encoder_18/Encoder-1st-NormalizationLayer-1/Reshape/shape/0:output:0Mmodel_6/transformer_encoder_18/Encoder-1st-NormalizationLayer-1/Prod:output:0Omodel_6/transformer_encoder_18/Encoder-1st-NormalizationLayer-1/Prod_1:output:0Xmodel_6/transformer_encoder_18/Encoder-1st-NormalizationLayer-1/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:�
Gmodel_6/transformer_encoder_18/Encoder-1st-NormalizationLayer-1/ReshapeReshape'model_6/transformer_encoder_18/Cast:y:0Vmodel_6/transformer_encoder_18/Encoder-1st-NormalizationLayer-1/Reshape/shape:output:0*
T0*8
_output_shapes&
$:"�������������������
Kmodel_6/transformer_encoder_18/Encoder-1st-NormalizationLayer-1/ones/packedPackMmodel_6/transformer_encoder_18/Encoder-1st-NormalizationLayer-1/Prod:output:0*
N*
T0*
_output_shapes
:�
Jmodel_6/transformer_encoder_18/Encoder-1st-NormalizationLayer-1/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
Dmodel_6/transformer_encoder_18/Encoder-1st-NormalizationLayer-1/onesFillTmodel_6/transformer_encoder_18/Encoder-1st-NormalizationLayer-1/ones/packed:output:0Smodel_6/transformer_encoder_18/Encoder-1st-NormalizationLayer-1/ones/Const:output:0*
T0*#
_output_shapes
:����������
Lmodel_6/transformer_encoder_18/Encoder-1st-NormalizationLayer-1/zeros/packedPackMmodel_6/transformer_encoder_18/Encoder-1st-NormalizationLayer-1/Prod:output:0*
N*
T0*
_output_shapes
:�
Kmodel_6/transformer_encoder_18/Encoder-1st-NormalizationLayer-1/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
Emodel_6/transformer_encoder_18/Encoder-1st-NormalizationLayer-1/zerosFillUmodel_6/transformer_encoder_18/Encoder-1st-NormalizationLayer-1/zeros/packed:output:0Tmodel_6/transformer_encoder_18/Encoder-1st-NormalizationLayer-1/zeros/Const:output:0*
T0*#
_output_shapes
:����������
Gmodel_6/transformer_encoder_18/Encoder-1st-NormalizationLayer-1/Const_2Const*
_output_shapes
: *
dtype0*
valueB �
Gmodel_6/transformer_encoder_18/Encoder-1st-NormalizationLayer-1/Const_3Const*
_output_shapes
: *
dtype0*
valueB �
Pmodel_6/transformer_encoder_18/Encoder-1st-NormalizationLayer-1/FusedBatchNormV3FusedBatchNormV3Pmodel_6/transformer_encoder_18/Encoder-1st-NormalizationLayer-1/Reshape:output:0Mmodel_6/transformer_encoder_18/Encoder-1st-NormalizationLayer-1/ones:output:0Nmodel_6/transformer_encoder_18/Encoder-1st-NormalizationLayer-1/zeros:output:0Pmodel_6/transformer_encoder_18/Encoder-1st-NormalizationLayer-1/Const_2:output:0Pmodel_6/transformer_encoder_18/Encoder-1st-NormalizationLayer-1/Const_3:output:0*
T0*
U0*x
_output_shapesf
d:"������������������:���������:���������:���������:���������:*
data_formatNCHW*
epsilon%o�:�
Imodel_6/transformer_encoder_18/Encoder-1st-NormalizationLayer-1/Reshape_1ReshapeTmodel_6/transformer_encoder_18/Encoder-1st-NormalizationLayer-1/FusedBatchNormV3:y:0Nmodel_6/transformer_encoder_18/Encoder-1st-NormalizationLayer-1/Shape:output:0*
T0*+
_output_shapes
:���������P	�
Rmodel_6/transformer_encoder_18/Encoder-1st-NormalizationLayer-1/mul/ReadVariableOpReadVariableOp[model_6_transformer_encoder_18_encoder_1st_normalizationlayer_1_mul_readvariableop_resource*
_output_shapes
:	*
dtype0�
Cmodel_6/transformer_encoder_18/Encoder-1st-NormalizationLayer-1/mulMulRmodel_6/transformer_encoder_18/Encoder-1st-NormalizationLayer-1/Reshape_1:output:0Zmodel_6/transformer_encoder_18/Encoder-1st-NormalizationLayer-1/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������P	�
Rmodel_6/transformer_encoder_18/Encoder-1st-NormalizationLayer-1/add/ReadVariableOpReadVariableOp[model_6_transformer_encoder_18_encoder_1st_normalizationlayer_1_add_readvariableop_resource*
_output_shapes
:	*
dtype0�
Cmodel_6/transformer_encoder_18/Encoder-1st-NormalizationLayer-1/addAddV2Gmodel_6/transformer_encoder_18/Encoder-1st-NormalizationLayer-1/mul:z:0Zmodel_6/transformer_encoder_18/Encoder-1st-NormalizationLayer-1/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������P	�
^model_6/transformer_encoder_18/Encoder-SelfAttentionLayer-1/query/einsum/Einsum/ReadVariableOpReadVariableOpgmodel_6_transformer_encoder_18_encoder_selfattentionlayer_1_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:	*
dtype0�
Omodel_6/transformer_encoder_18/Encoder-SelfAttentionLayer-1/query/einsum/EinsumEinsumGmodel_6/transformer_encoder_18/Encoder-1st-NormalizationLayer-1/add:z:0fmodel_6/transformer_encoder_18/Encoder-SelfAttentionLayer-1/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������P*
equationabc,cde->abde�
Tmodel_6/transformer_encoder_18/Encoder-SelfAttentionLayer-1/query/add/ReadVariableOpReadVariableOp]model_6_transformer_encoder_18_encoder_selfattentionlayer_1_query_add_readvariableop_resource*
_output_shapes

:*
dtype0�
Emodel_6/transformer_encoder_18/Encoder-SelfAttentionLayer-1/query/addAddV2Xmodel_6/transformer_encoder_18/Encoder-SelfAttentionLayer-1/query/einsum/Einsum:output:0\model_6/transformer_encoder_18/Encoder-SelfAttentionLayer-1/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������P�
\model_6/transformer_encoder_18/Encoder-SelfAttentionLayer-1/key/einsum/Einsum/ReadVariableOpReadVariableOpemodel_6_transformer_encoder_18_encoder_selfattentionlayer_1_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:	*
dtype0�
Mmodel_6/transformer_encoder_18/Encoder-SelfAttentionLayer-1/key/einsum/EinsumEinsumGmodel_6/transformer_encoder_18/Encoder-1st-NormalizationLayer-1/add:z:0dmodel_6/transformer_encoder_18/Encoder-SelfAttentionLayer-1/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������P*
equationabc,cde->abde�
Rmodel_6/transformer_encoder_18/Encoder-SelfAttentionLayer-1/key/add/ReadVariableOpReadVariableOp[model_6_transformer_encoder_18_encoder_selfattentionlayer_1_key_add_readvariableop_resource*
_output_shapes

:*
dtype0�
Cmodel_6/transformer_encoder_18/Encoder-SelfAttentionLayer-1/key/addAddV2Vmodel_6/transformer_encoder_18/Encoder-SelfAttentionLayer-1/key/einsum/Einsum:output:0Zmodel_6/transformer_encoder_18/Encoder-SelfAttentionLayer-1/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������P�
^model_6/transformer_encoder_18/Encoder-SelfAttentionLayer-1/value/einsum/Einsum/ReadVariableOpReadVariableOpgmodel_6_transformer_encoder_18_encoder_selfattentionlayer_1_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:	*
dtype0�
Omodel_6/transformer_encoder_18/Encoder-SelfAttentionLayer-1/value/einsum/EinsumEinsumGmodel_6/transformer_encoder_18/Encoder-1st-NormalizationLayer-1/add:z:0fmodel_6/transformer_encoder_18/Encoder-SelfAttentionLayer-1/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������P*
equationabc,cde->abde�
Tmodel_6/transformer_encoder_18/Encoder-SelfAttentionLayer-1/value/add/ReadVariableOpReadVariableOp]model_6_transformer_encoder_18_encoder_selfattentionlayer_1_value_add_readvariableop_resource*
_output_shapes

:*
dtype0�
Emodel_6/transformer_encoder_18/Encoder-SelfAttentionLayer-1/value/addAddV2Xmodel_6/transformer_encoder_18/Encoder-SelfAttentionLayer-1/value/einsum/Einsum:output:0\model_6/transformer_encoder_18/Encoder-SelfAttentionLayer-1/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������P�
Amodel_6/transformer_encoder_18/Encoder-SelfAttentionLayer-1/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *:�?�
?model_6/transformer_encoder_18/Encoder-SelfAttentionLayer-1/MulMulImodel_6/transformer_encoder_18/Encoder-SelfAttentionLayer-1/query/add:z:0Jmodel_6/transformer_encoder_18/Encoder-SelfAttentionLayer-1/Mul/y:output:0*
T0*/
_output_shapes
:���������P�
Imodel_6/transformer_encoder_18/Encoder-SelfAttentionLayer-1/einsum/EinsumEinsumGmodel_6/transformer_encoder_18/Encoder-SelfAttentionLayer-1/key/add:z:0Cmodel_6/transformer_encoder_18/Encoder-SelfAttentionLayer-1/Mul:z:0*
N*
T0*/
_output_shapes
:���������PP*
equationaecd,abcd->acbe�
Kmodel_6/transformer_encoder_18/Encoder-SelfAttentionLayer-1/softmax/SoftmaxSoftmaxRmodel_6/transformer_encoder_18/Encoder-SelfAttentionLayer-1/einsum/Einsum:output:0*
T0*/
_output_shapes
:���������PP�
Lmodel_6/transformer_encoder_18/Encoder-SelfAttentionLayer-1/dropout/IdentityIdentityUmodel_6/transformer_encoder_18/Encoder-SelfAttentionLayer-1/softmax/Softmax:softmax:0*
T0*/
_output_shapes
:���������PP�
Kmodel_6/transformer_encoder_18/Encoder-SelfAttentionLayer-1/einsum_1/EinsumEinsumUmodel_6/transformer_encoder_18/Encoder-SelfAttentionLayer-1/dropout/Identity:output:0Imodel_6/transformer_encoder_18/Encoder-SelfAttentionLayer-1/value/add:z:0*
N*
T0*/
_output_shapes
:���������P*
equationacbe,aecd->abcd�
imodel_6/transformer_encoder_18/Encoder-SelfAttentionLayer-1/attention_output/einsum/Einsum/ReadVariableOpReadVariableOprmodel_6_transformer_encoder_18_encoder_selfattentionlayer_1_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:	*
dtype0�
Zmodel_6/transformer_encoder_18/Encoder-SelfAttentionLayer-1/attention_output/einsum/EinsumEinsumTmodel_6/transformer_encoder_18/Encoder-SelfAttentionLayer-1/einsum_1/Einsum:output:0qmodel_6/transformer_encoder_18/Encoder-SelfAttentionLayer-1/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:���������P	*
equationabcd,cde->abe�
_model_6/transformer_encoder_18/Encoder-SelfAttentionLayer-1/attention_output/add/ReadVariableOpReadVariableOphmodel_6_transformer_encoder_18_encoder_selfattentionlayer_1_attention_output_add_readvariableop_resource*
_output_shapes
:	*
dtype0�
Pmodel_6/transformer_encoder_18/Encoder-SelfAttentionLayer-1/attention_output/addAddV2cmodel_6/transformer_encoder_18/Encoder-SelfAttentionLayer-1/attention_output/einsum/Einsum:output:0gmodel_6/transformer_encoder_18/Encoder-SelfAttentionLayer-1/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������P	�
>model_6/transformer_encoder_18/Encoder-1st-AdditionLayer-1/addAddV2'model_6/transformer_encoder_18/Cast:y:0Tmodel_6/transformer_encoder_18/Encoder-SelfAttentionLayer-1/attention_output/add:z:0*
T0*+
_output_shapes
:���������P	�
Emodel_6/transformer_encoder_18/Encoder-2nd-NormalizationLayer-1/ShapeShapeBmodel_6/transformer_encoder_18/Encoder-1st-AdditionLayer-1/add:z:0*
T0*
_output_shapes
::���
Smodel_6/transformer_encoder_18/Encoder-2nd-NormalizationLayer-1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
Umodel_6/transformer_encoder_18/Encoder-2nd-NormalizationLayer-1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
Umodel_6/transformer_encoder_18/Encoder-2nd-NormalizationLayer-1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
Mmodel_6/transformer_encoder_18/Encoder-2nd-NormalizationLayer-1/strided_sliceStridedSliceNmodel_6/transformer_encoder_18/Encoder-2nd-NormalizationLayer-1/Shape:output:0\model_6/transformer_encoder_18/Encoder-2nd-NormalizationLayer-1/strided_slice/stack:output:0^model_6/transformer_encoder_18/Encoder-2nd-NormalizationLayer-1/strided_slice/stack_1:output:0^model_6/transformer_encoder_18/Encoder-2nd-NormalizationLayer-1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask�
Emodel_6/transformer_encoder_18/Encoder-2nd-NormalizationLayer-1/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
Dmodel_6/transformer_encoder_18/Encoder-2nd-NormalizationLayer-1/ProdProdVmodel_6/transformer_encoder_18/Encoder-2nd-NormalizationLayer-1/strided_slice:output:0Nmodel_6/transformer_encoder_18/Encoder-2nd-NormalizationLayer-1/Const:output:0*
T0*
_output_shapes
: �
Umodel_6/transformer_encoder_18/Encoder-2nd-NormalizationLayer-1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:�
Wmodel_6/transformer_encoder_18/Encoder-2nd-NormalizationLayer-1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: �
Wmodel_6/transformer_encoder_18/Encoder-2nd-NormalizationLayer-1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
Omodel_6/transformer_encoder_18/Encoder-2nd-NormalizationLayer-1/strided_slice_1StridedSliceNmodel_6/transformer_encoder_18/Encoder-2nd-NormalizationLayer-1/Shape:output:0^model_6/transformer_encoder_18/Encoder-2nd-NormalizationLayer-1/strided_slice_1/stack:output:0`model_6/transformer_encoder_18/Encoder-2nd-NormalizationLayer-1/strided_slice_1/stack_1:output:0`model_6/transformer_encoder_18/Encoder-2nd-NormalizationLayer-1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask�
Gmodel_6/transformer_encoder_18/Encoder-2nd-NormalizationLayer-1/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
Fmodel_6/transformer_encoder_18/Encoder-2nd-NormalizationLayer-1/Prod_1ProdXmodel_6/transformer_encoder_18/Encoder-2nd-NormalizationLayer-1/strided_slice_1:output:0Pmodel_6/transformer_encoder_18/Encoder-2nd-NormalizationLayer-1/Const_1:output:0*
T0*
_output_shapes
: �
Omodel_6/transformer_encoder_18/Encoder-2nd-NormalizationLayer-1/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :�
Omodel_6/transformer_encoder_18/Encoder-2nd-NormalizationLayer-1/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :�
Mmodel_6/transformer_encoder_18/Encoder-2nd-NormalizationLayer-1/Reshape/shapePackXmodel_6/transformer_encoder_18/Encoder-2nd-NormalizationLayer-1/Reshape/shape/0:output:0Mmodel_6/transformer_encoder_18/Encoder-2nd-NormalizationLayer-1/Prod:output:0Omodel_6/transformer_encoder_18/Encoder-2nd-NormalizationLayer-1/Prod_1:output:0Xmodel_6/transformer_encoder_18/Encoder-2nd-NormalizationLayer-1/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:�
Gmodel_6/transformer_encoder_18/Encoder-2nd-NormalizationLayer-1/ReshapeReshapeBmodel_6/transformer_encoder_18/Encoder-1st-AdditionLayer-1/add:z:0Vmodel_6/transformer_encoder_18/Encoder-2nd-NormalizationLayer-1/Reshape/shape:output:0*
T0*8
_output_shapes&
$:"�������������������
Kmodel_6/transformer_encoder_18/Encoder-2nd-NormalizationLayer-1/ones/packedPackMmodel_6/transformer_encoder_18/Encoder-2nd-NormalizationLayer-1/Prod:output:0*
N*
T0*
_output_shapes
:�
Jmodel_6/transformer_encoder_18/Encoder-2nd-NormalizationLayer-1/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
Dmodel_6/transformer_encoder_18/Encoder-2nd-NormalizationLayer-1/onesFillTmodel_6/transformer_encoder_18/Encoder-2nd-NormalizationLayer-1/ones/packed:output:0Smodel_6/transformer_encoder_18/Encoder-2nd-NormalizationLayer-1/ones/Const:output:0*
T0*#
_output_shapes
:����������
Lmodel_6/transformer_encoder_18/Encoder-2nd-NormalizationLayer-1/zeros/packedPackMmodel_6/transformer_encoder_18/Encoder-2nd-NormalizationLayer-1/Prod:output:0*
N*
T0*
_output_shapes
:�
Kmodel_6/transformer_encoder_18/Encoder-2nd-NormalizationLayer-1/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
Emodel_6/transformer_encoder_18/Encoder-2nd-NormalizationLayer-1/zerosFillUmodel_6/transformer_encoder_18/Encoder-2nd-NormalizationLayer-1/zeros/packed:output:0Tmodel_6/transformer_encoder_18/Encoder-2nd-NormalizationLayer-1/zeros/Const:output:0*
T0*#
_output_shapes
:����������
Gmodel_6/transformer_encoder_18/Encoder-2nd-NormalizationLayer-1/Const_2Const*
_output_shapes
: *
dtype0*
valueB �
Gmodel_6/transformer_encoder_18/Encoder-2nd-NormalizationLayer-1/Const_3Const*
_output_shapes
: *
dtype0*
valueB �
Pmodel_6/transformer_encoder_18/Encoder-2nd-NormalizationLayer-1/FusedBatchNormV3FusedBatchNormV3Pmodel_6/transformer_encoder_18/Encoder-2nd-NormalizationLayer-1/Reshape:output:0Mmodel_6/transformer_encoder_18/Encoder-2nd-NormalizationLayer-1/ones:output:0Nmodel_6/transformer_encoder_18/Encoder-2nd-NormalizationLayer-1/zeros:output:0Pmodel_6/transformer_encoder_18/Encoder-2nd-NormalizationLayer-1/Const_2:output:0Pmodel_6/transformer_encoder_18/Encoder-2nd-NormalizationLayer-1/Const_3:output:0*
T0*
U0*x
_output_shapesf
d:"������������������:���������:���������:���������:���������:*
data_formatNCHW*
epsilon%o�:�
Imodel_6/transformer_encoder_18/Encoder-2nd-NormalizationLayer-1/Reshape_1ReshapeTmodel_6/transformer_encoder_18/Encoder-2nd-NormalizationLayer-1/FusedBatchNormV3:y:0Nmodel_6/transformer_encoder_18/Encoder-2nd-NormalizationLayer-1/Shape:output:0*
T0*+
_output_shapes
:���������P	�
Rmodel_6/transformer_encoder_18/Encoder-2nd-NormalizationLayer-1/mul/ReadVariableOpReadVariableOp[model_6_transformer_encoder_18_encoder_2nd_normalizationlayer_1_mul_readvariableop_resource*
_output_shapes
:	*
dtype0�
Cmodel_6/transformer_encoder_18/Encoder-2nd-NormalizationLayer-1/mulMulRmodel_6/transformer_encoder_18/Encoder-2nd-NormalizationLayer-1/Reshape_1:output:0Zmodel_6/transformer_encoder_18/Encoder-2nd-NormalizationLayer-1/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������P	�
Rmodel_6/transformer_encoder_18/Encoder-2nd-NormalizationLayer-1/add/ReadVariableOpReadVariableOp[model_6_transformer_encoder_18_encoder_2nd_normalizationlayer_1_add_readvariableop_resource*
_output_shapes
:	*
dtype0�
Cmodel_6/transformer_encoder_18/Encoder-2nd-NormalizationLayer-1/addAddV2Gmodel_6/transformer_encoder_18/Encoder-2nd-NormalizationLayer-1/mul:z:0Zmodel_6/transformer_encoder_18/Encoder-2nd-NormalizationLayer-1/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������P	�
Tmodel_6/transformer_encoder_18/Encoder-FeedForwardLayer_1_1/Tensordot/ReadVariableOpReadVariableOp]model_6_transformer_encoder_18_encoder_feedforwardlayer_1_1_tensordot_readvariableop_resource*
_output_shapes

:		*
dtype0�
Jmodel_6/transformer_encoder_18/Encoder-FeedForwardLayer_1_1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:�
Jmodel_6/transformer_encoder_18/Encoder-FeedForwardLayer_1_1/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       �
Kmodel_6/transformer_encoder_18/Encoder-FeedForwardLayer_1_1/Tensordot/ShapeShapeGmodel_6/transformer_encoder_18/Encoder-2nd-NormalizationLayer-1/add:z:0*
T0*
_output_shapes
::���
Smodel_6/transformer_encoder_18/Encoder-FeedForwardLayer_1_1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Nmodel_6/transformer_encoder_18/Encoder-FeedForwardLayer_1_1/Tensordot/GatherV2GatherV2Tmodel_6/transformer_encoder_18/Encoder-FeedForwardLayer_1_1/Tensordot/Shape:output:0Smodel_6/transformer_encoder_18/Encoder-FeedForwardLayer_1_1/Tensordot/free:output:0\model_6/transformer_encoder_18/Encoder-FeedForwardLayer_1_1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
Umodel_6/transformer_encoder_18/Encoder-FeedForwardLayer_1_1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Pmodel_6/transformer_encoder_18/Encoder-FeedForwardLayer_1_1/Tensordot/GatherV2_1GatherV2Tmodel_6/transformer_encoder_18/Encoder-FeedForwardLayer_1_1/Tensordot/Shape:output:0Smodel_6/transformer_encoder_18/Encoder-FeedForwardLayer_1_1/Tensordot/axes:output:0^model_6/transformer_encoder_18/Encoder-FeedForwardLayer_1_1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
Kmodel_6/transformer_encoder_18/Encoder-FeedForwardLayer_1_1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
Jmodel_6/transformer_encoder_18/Encoder-FeedForwardLayer_1_1/Tensordot/ProdProdWmodel_6/transformer_encoder_18/Encoder-FeedForwardLayer_1_1/Tensordot/GatherV2:output:0Tmodel_6/transformer_encoder_18/Encoder-FeedForwardLayer_1_1/Tensordot/Const:output:0*
T0*
_output_shapes
: �
Mmodel_6/transformer_encoder_18/Encoder-FeedForwardLayer_1_1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
Lmodel_6/transformer_encoder_18/Encoder-FeedForwardLayer_1_1/Tensordot/Prod_1ProdYmodel_6/transformer_encoder_18/Encoder-FeedForwardLayer_1_1/Tensordot/GatherV2_1:output:0Vmodel_6/transformer_encoder_18/Encoder-FeedForwardLayer_1_1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: �
Qmodel_6/transformer_encoder_18/Encoder-FeedForwardLayer_1_1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Lmodel_6/transformer_encoder_18/Encoder-FeedForwardLayer_1_1/Tensordot/concatConcatV2Smodel_6/transformer_encoder_18/Encoder-FeedForwardLayer_1_1/Tensordot/free:output:0Smodel_6/transformer_encoder_18/Encoder-FeedForwardLayer_1_1/Tensordot/axes:output:0Zmodel_6/transformer_encoder_18/Encoder-FeedForwardLayer_1_1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
Kmodel_6/transformer_encoder_18/Encoder-FeedForwardLayer_1_1/Tensordot/stackPackSmodel_6/transformer_encoder_18/Encoder-FeedForwardLayer_1_1/Tensordot/Prod:output:0Umodel_6/transformer_encoder_18/Encoder-FeedForwardLayer_1_1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
Omodel_6/transformer_encoder_18/Encoder-FeedForwardLayer_1_1/Tensordot/transpose	TransposeGmodel_6/transformer_encoder_18/Encoder-2nd-NormalizationLayer-1/add:z:0Umodel_6/transformer_encoder_18/Encoder-FeedForwardLayer_1_1/Tensordot/concat:output:0*
T0*+
_output_shapes
:���������P	�
Mmodel_6/transformer_encoder_18/Encoder-FeedForwardLayer_1_1/Tensordot/ReshapeReshapeSmodel_6/transformer_encoder_18/Encoder-FeedForwardLayer_1_1/Tensordot/transpose:y:0Tmodel_6/transformer_encoder_18/Encoder-FeedForwardLayer_1_1/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
Lmodel_6/transformer_encoder_18/Encoder-FeedForwardLayer_1_1/Tensordot/MatMulMatMulVmodel_6/transformer_encoder_18/Encoder-FeedForwardLayer_1_1/Tensordot/Reshape:output:0\model_6/transformer_encoder_18/Encoder-FeedForwardLayer_1_1/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������	�
Mmodel_6/transformer_encoder_18/Encoder-FeedForwardLayer_1_1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:	�
Smodel_6/transformer_encoder_18/Encoder-FeedForwardLayer_1_1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Nmodel_6/transformer_encoder_18/Encoder-FeedForwardLayer_1_1/Tensordot/concat_1ConcatV2Wmodel_6/transformer_encoder_18/Encoder-FeedForwardLayer_1_1/Tensordot/GatherV2:output:0Vmodel_6/transformer_encoder_18/Encoder-FeedForwardLayer_1_1/Tensordot/Const_2:output:0\model_6/transformer_encoder_18/Encoder-FeedForwardLayer_1_1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
Emodel_6/transformer_encoder_18/Encoder-FeedForwardLayer_1_1/TensordotReshapeVmodel_6/transformer_encoder_18/Encoder-FeedForwardLayer_1_1/Tensordot/MatMul:product:0Wmodel_6/transformer_encoder_18/Encoder-FeedForwardLayer_1_1/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������P	�
Rmodel_6/transformer_encoder_18/Encoder-FeedForwardLayer_1_1/BiasAdd/ReadVariableOpReadVariableOp[model_6_transformer_encoder_18_encoder_feedforwardlayer_1_1_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype0�
Cmodel_6/transformer_encoder_18/Encoder-FeedForwardLayer_1_1/BiasAddBiasAddNmodel_6/transformer_encoder_18/Encoder-FeedForwardLayer_1_1/Tensordot:output:0Zmodel_6/transformer_encoder_18/Encoder-FeedForwardLayer_1_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������P	�
Fmodel_6/transformer_encoder_18/Encoder-FeedForwardLayer_1_1/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
Dmodel_6/transformer_encoder_18/Encoder-FeedForwardLayer_1_1/Gelu/mulMulOmodel_6/transformer_encoder_18/Encoder-FeedForwardLayer_1_1/Gelu/mul/x:output:0Lmodel_6/transformer_encoder_18/Encoder-FeedForwardLayer_1_1/BiasAdd:output:0*
T0*+
_output_shapes
:���������P	�
Gmodel_6/transformer_encoder_18/Encoder-FeedForwardLayer_1_1/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *��?�
Hmodel_6/transformer_encoder_18/Encoder-FeedForwardLayer_1_1/Gelu/truedivRealDivLmodel_6/transformer_encoder_18/Encoder-FeedForwardLayer_1_1/BiasAdd:output:0Pmodel_6/transformer_encoder_18/Encoder-FeedForwardLayer_1_1/Gelu/Cast/x:output:0*
T0*+
_output_shapes
:���������P	�
Dmodel_6/transformer_encoder_18/Encoder-FeedForwardLayer_1_1/Gelu/ErfErfLmodel_6/transformer_encoder_18/Encoder-FeedForwardLayer_1_1/Gelu/truediv:z:0*
T0*+
_output_shapes
:���������P	�
Fmodel_6/transformer_encoder_18/Encoder-FeedForwardLayer_1_1/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
Dmodel_6/transformer_encoder_18/Encoder-FeedForwardLayer_1_1/Gelu/addAddV2Omodel_6/transformer_encoder_18/Encoder-FeedForwardLayer_1_1/Gelu/add/x:output:0Hmodel_6/transformer_encoder_18/Encoder-FeedForwardLayer_1_1/Gelu/Erf:y:0*
T0*+
_output_shapes
:���������P	�
Fmodel_6/transformer_encoder_18/Encoder-FeedForwardLayer_1_1/Gelu/mul_1MulHmodel_6/transformer_encoder_18/Encoder-FeedForwardLayer_1_1/Gelu/mul:z:0Hmodel_6/transformer_encoder_18/Encoder-FeedForwardLayer_1_1/Gelu/add:z:0*
T0*+
_output_shapes
:���������P	�
Tmodel_6/transformer_encoder_18/Encoder-FeedForwardLayer_2_1/Tensordot/ReadVariableOpReadVariableOp]model_6_transformer_encoder_18_encoder_feedforwardlayer_2_1_tensordot_readvariableop_resource*
_output_shapes

:		*
dtype0�
Jmodel_6/transformer_encoder_18/Encoder-FeedForwardLayer_2_1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:�
Jmodel_6/transformer_encoder_18/Encoder-FeedForwardLayer_2_1/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       �
Kmodel_6/transformer_encoder_18/Encoder-FeedForwardLayer_2_1/Tensordot/ShapeShapeJmodel_6/transformer_encoder_18/Encoder-FeedForwardLayer_1_1/Gelu/mul_1:z:0*
T0*
_output_shapes
::���
Smodel_6/transformer_encoder_18/Encoder-FeedForwardLayer_2_1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Nmodel_6/transformer_encoder_18/Encoder-FeedForwardLayer_2_1/Tensordot/GatherV2GatherV2Tmodel_6/transformer_encoder_18/Encoder-FeedForwardLayer_2_1/Tensordot/Shape:output:0Smodel_6/transformer_encoder_18/Encoder-FeedForwardLayer_2_1/Tensordot/free:output:0\model_6/transformer_encoder_18/Encoder-FeedForwardLayer_2_1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
Umodel_6/transformer_encoder_18/Encoder-FeedForwardLayer_2_1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Pmodel_6/transformer_encoder_18/Encoder-FeedForwardLayer_2_1/Tensordot/GatherV2_1GatherV2Tmodel_6/transformer_encoder_18/Encoder-FeedForwardLayer_2_1/Tensordot/Shape:output:0Smodel_6/transformer_encoder_18/Encoder-FeedForwardLayer_2_1/Tensordot/axes:output:0^model_6/transformer_encoder_18/Encoder-FeedForwardLayer_2_1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
Kmodel_6/transformer_encoder_18/Encoder-FeedForwardLayer_2_1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
Jmodel_6/transformer_encoder_18/Encoder-FeedForwardLayer_2_1/Tensordot/ProdProdWmodel_6/transformer_encoder_18/Encoder-FeedForwardLayer_2_1/Tensordot/GatherV2:output:0Tmodel_6/transformer_encoder_18/Encoder-FeedForwardLayer_2_1/Tensordot/Const:output:0*
T0*
_output_shapes
: �
Mmodel_6/transformer_encoder_18/Encoder-FeedForwardLayer_2_1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
Lmodel_6/transformer_encoder_18/Encoder-FeedForwardLayer_2_1/Tensordot/Prod_1ProdYmodel_6/transformer_encoder_18/Encoder-FeedForwardLayer_2_1/Tensordot/GatherV2_1:output:0Vmodel_6/transformer_encoder_18/Encoder-FeedForwardLayer_2_1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: �
Qmodel_6/transformer_encoder_18/Encoder-FeedForwardLayer_2_1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Lmodel_6/transformer_encoder_18/Encoder-FeedForwardLayer_2_1/Tensordot/concatConcatV2Smodel_6/transformer_encoder_18/Encoder-FeedForwardLayer_2_1/Tensordot/free:output:0Smodel_6/transformer_encoder_18/Encoder-FeedForwardLayer_2_1/Tensordot/axes:output:0Zmodel_6/transformer_encoder_18/Encoder-FeedForwardLayer_2_1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
Kmodel_6/transformer_encoder_18/Encoder-FeedForwardLayer_2_1/Tensordot/stackPackSmodel_6/transformer_encoder_18/Encoder-FeedForwardLayer_2_1/Tensordot/Prod:output:0Umodel_6/transformer_encoder_18/Encoder-FeedForwardLayer_2_1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
Omodel_6/transformer_encoder_18/Encoder-FeedForwardLayer_2_1/Tensordot/transpose	TransposeJmodel_6/transformer_encoder_18/Encoder-FeedForwardLayer_1_1/Gelu/mul_1:z:0Umodel_6/transformer_encoder_18/Encoder-FeedForwardLayer_2_1/Tensordot/concat:output:0*
T0*+
_output_shapes
:���������P	�
Mmodel_6/transformer_encoder_18/Encoder-FeedForwardLayer_2_1/Tensordot/ReshapeReshapeSmodel_6/transformer_encoder_18/Encoder-FeedForwardLayer_2_1/Tensordot/transpose:y:0Tmodel_6/transformer_encoder_18/Encoder-FeedForwardLayer_2_1/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
Lmodel_6/transformer_encoder_18/Encoder-FeedForwardLayer_2_1/Tensordot/MatMulMatMulVmodel_6/transformer_encoder_18/Encoder-FeedForwardLayer_2_1/Tensordot/Reshape:output:0\model_6/transformer_encoder_18/Encoder-FeedForwardLayer_2_1/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������	�
Mmodel_6/transformer_encoder_18/Encoder-FeedForwardLayer_2_1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:	�
Smodel_6/transformer_encoder_18/Encoder-FeedForwardLayer_2_1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Nmodel_6/transformer_encoder_18/Encoder-FeedForwardLayer_2_1/Tensordot/concat_1ConcatV2Wmodel_6/transformer_encoder_18/Encoder-FeedForwardLayer_2_1/Tensordot/GatherV2:output:0Vmodel_6/transformer_encoder_18/Encoder-FeedForwardLayer_2_1/Tensordot/Const_2:output:0\model_6/transformer_encoder_18/Encoder-FeedForwardLayer_2_1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
Emodel_6/transformer_encoder_18/Encoder-FeedForwardLayer_2_1/TensordotReshapeVmodel_6/transformer_encoder_18/Encoder-FeedForwardLayer_2_1/Tensordot/MatMul:product:0Wmodel_6/transformer_encoder_18/Encoder-FeedForwardLayer_2_1/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������P	�
Rmodel_6/transformer_encoder_18/Encoder-FeedForwardLayer_2_1/BiasAdd/ReadVariableOpReadVariableOp[model_6_transformer_encoder_18_encoder_feedforwardlayer_2_1_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype0�
Cmodel_6/transformer_encoder_18/Encoder-FeedForwardLayer_2_1/BiasAddBiasAddNmodel_6/transformer_encoder_18/Encoder-FeedForwardLayer_2_1/Tensordot:output:0Zmodel_6/transformer_encoder_18/Encoder-FeedForwardLayer_2_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������P	�
2model_6/transformer_encoder_18/dropout_18/IdentityIdentityLmodel_6/transformer_encoder_18/Encoder-FeedForwardLayer_2_1/BiasAdd:output:0*
T0*+
_output_shapes
:���������P	�
>model_6/transformer_encoder_18/Encoder-2nd-AdditionLayer-1/addAddV2Bmodel_6/transformer_encoder_18/Encoder-1st-AdditionLayer-1/add:z:0;model_6/transformer_encoder_18/dropout_18/Identity:output:0*
T0*+
_output_shapes
:���������P	�
Emodel_6/transformer_encoder_19/Encoder-1st-NormalizationLayer-2/ShapeShapeBmodel_6/transformer_encoder_18/Encoder-2nd-AdditionLayer-1/add:z:0*
T0*
_output_shapes
::���
Smodel_6/transformer_encoder_19/Encoder-1st-NormalizationLayer-2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
Umodel_6/transformer_encoder_19/Encoder-1st-NormalizationLayer-2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
Umodel_6/transformer_encoder_19/Encoder-1st-NormalizationLayer-2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
Mmodel_6/transformer_encoder_19/Encoder-1st-NormalizationLayer-2/strided_sliceStridedSliceNmodel_6/transformer_encoder_19/Encoder-1st-NormalizationLayer-2/Shape:output:0\model_6/transformer_encoder_19/Encoder-1st-NormalizationLayer-2/strided_slice/stack:output:0^model_6/transformer_encoder_19/Encoder-1st-NormalizationLayer-2/strided_slice/stack_1:output:0^model_6/transformer_encoder_19/Encoder-1st-NormalizationLayer-2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask�
Emodel_6/transformer_encoder_19/Encoder-1st-NormalizationLayer-2/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
Dmodel_6/transformer_encoder_19/Encoder-1st-NormalizationLayer-2/ProdProdVmodel_6/transformer_encoder_19/Encoder-1st-NormalizationLayer-2/strided_slice:output:0Nmodel_6/transformer_encoder_19/Encoder-1st-NormalizationLayer-2/Const:output:0*
T0*
_output_shapes
: �
Umodel_6/transformer_encoder_19/Encoder-1st-NormalizationLayer-2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:�
Wmodel_6/transformer_encoder_19/Encoder-1st-NormalizationLayer-2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: �
Wmodel_6/transformer_encoder_19/Encoder-1st-NormalizationLayer-2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
Omodel_6/transformer_encoder_19/Encoder-1st-NormalizationLayer-2/strided_slice_1StridedSliceNmodel_6/transformer_encoder_19/Encoder-1st-NormalizationLayer-2/Shape:output:0^model_6/transformer_encoder_19/Encoder-1st-NormalizationLayer-2/strided_slice_1/stack:output:0`model_6/transformer_encoder_19/Encoder-1st-NormalizationLayer-2/strided_slice_1/stack_1:output:0`model_6/transformer_encoder_19/Encoder-1st-NormalizationLayer-2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask�
Gmodel_6/transformer_encoder_19/Encoder-1st-NormalizationLayer-2/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
Fmodel_6/transformer_encoder_19/Encoder-1st-NormalizationLayer-2/Prod_1ProdXmodel_6/transformer_encoder_19/Encoder-1st-NormalizationLayer-2/strided_slice_1:output:0Pmodel_6/transformer_encoder_19/Encoder-1st-NormalizationLayer-2/Const_1:output:0*
T0*
_output_shapes
: �
Omodel_6/transformer_encoder_19/Encoder-1st-NormalizationLayer-2/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :�
Omodel_6/transformer_encoder_19/Encoder-1st-NormalizationLayer-2/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :�
Mmodel_6/transformer_encoder_19/Encoder-1st-NormalizationLayer-2/Reshape/shapePackXmodel_6/transformer_encoder_19/Encoder-1st-NormalizationLayer-2/Reshape/shape/0:output:0Mmodel_6/transformer_encoder_19/Encoder-1st-NormalizationLayer-2/Prod:output:0Omodel_6/transformer_encoder_19/Encoder-1st-NormalizationLayer-2/Prod_1:output:0Xmodel_6/transformer_encoder_19/Encoder-1st-NormalizationLayer-2/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:�
Gmodel_6/transformer_encoder_19/Encoder-1st-NormalizationLayer-2/ReshapeReshapeBmodel_6/transformer_encoder_18/Encoder-2nd-AdditionLayer-1/add:z:0Vmodel_6/transformer_encoder_19/Encoder-1st-NormalizationLayer-2/Reshape/shape:output:0*
T0*8
_output_shapes&
$:"�������������������
Kmodel_6/transformer_encoder_19/Encoder-1st-NormalizationLayer-2/ones/packedPackMmodel_6/transformer_encoder_19/Encoder-1st-NormalizationLayer-2/Prod:output:0*
N*
T0*
_output_shapes
:�
Jmodel_6/transformer_encoder_19/Encoder-1st-NormalizationLayer-2/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
Dmodel_6/transformer_encoder_19/Encoder-1st-NormalizationLayer-2/onesFillTmodel_6/transformer_encoder_19/Encoder-1st-NormalizationLayer-2/ones/packed:output:0Smodel_6/transformer_encoder_19/Encoder-1st-NormalizationLayer-2/ones/Const:output:0*
T0*#
_output_shapes
:����������
Lmodel_6/transformer_encoder_19/Encoder-1st-NormalizationLayer-2/zeros/packedPackMmodel_6/transformer_encoder_19/Encoder-1st-NormalizationLayer-2/Prod:output:0*
N*
T0*
_output_shapes
:�
Kmodel_6/transformer_encoder_19/Encoder-1st-NormalizationLayer-2/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
Emodel_6/transformer_encoder_19/Encoder-1st-NormalizationLayer-2/zerosFillUmodel_6/transformer_encoder_19/Encoder-1st-NormalizationLayer-2/zeros/packed:output:0Tmodel_6/transformer_encoder_19/Encoder-1st-NormalizationLayer-2/zeros/Const:output:0*
T0*#
_output_shapes
:����������
Gmodel_6/transformer_encoder_19/Encoder-1st-NormalizationLayer-2/Const_2Const*
_output_shapes
: *
dtype0*
valueB �
Gmodel_6/transformer_encoder_19/Encoder-1st-NormalizationLayer-2/Const_3Const*
_output_shapes
: *
dtype0*
valueB �
Pmodel_6/transformer_encoder_19/Encoder-1st-NormalizationLayer-2/FusedBatchNormV3FusedBatchNormV3Pmodel_6/transformer_encoder_19/Encoder-1st-NormalizationLayer-2/Reshape:output:0Mmodel_6/transformer_encoder_19/Encoder-1st-NormalizationLayer-2/ones:output:0Nmodel_6/transformer_encoder_19/Encoder-1st-NormalizationLayer-2/zeros:output:0Pmodel_6/transformer_encoder_19/Encoder-1st-NormalizationLayer-2/Const_2:output:0Pmodel_6/transformer_encoder_19/Encoder-1st-NormalizationLayer-2/Const_3:output:0*
T0*
U0*x
_output_shapesf
d:"������������������:���������:���������:���������:���������:*
data_formatNCHW*
epsilon%o�:�
Imodel_6/transformer_encoder_19/Encoder-1st-NormalizationLayer-2/Reshape_1ReshapeTmodel_6/transformer_encoder_19/Encoder-1st-NormalizationLayer-2/FusedBatchNormV3:y:0Nmodel_6/transformer_encoder_19/Encoder-1st-NormalizationLayer-2/Shape:output:0*
T0*+
_output_shapes
:���������P	�
Rmodel_6/transformer_encoder_19/Encoder-1st-NormalizationLayer-2/mul/ReadVariableOpReadVariableOp[model_6_transformer_encoder_19_encoder_1st_normalizationlayer_2_mul_readvariableop_resource*
_output_shapes
:	*
dtype0�
Cmodel_6/transformer_encoder_19/Encoder-1st-NormalizationLayer-2/mulMulRmodel_6/transformer_encoder_19/Encoder-1st-NormalizationLayer-2/Reshape_1:output:0Zmodel_6/transformer_encoder_19/Encoder-1st-NormalizationLayer-2/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������P	�
Rmodel_6/transformer_encoder_19/Encoder-1st-NormalizationLayer-2/add/ReadVariableOpReadVariableOp[model_6_transformer_encoder_19_encoder_1st_normalizationlayer_2_add_readvariableop_resource*
_output_shapes
:	*
dtype0�
Cmodel_6/transformer_encoder_19/Encoder-1st-NormalizationLayer-2/addAddV2Gmodel_6/transformer_encoder_19/Encoder-1st-NormalizationLayer-2/mul:z:0Zmodel_6/transformer_encoder_19/Encoder-1st-NormalizationLayer-2/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������P	�
^model_6/transformer_encoder_19/Encoder-SelfAttentionLayer-2/query/einsum/Einsum/ReadVariableOpReadVariableOpgmodel_6_transformer_encoder_19_encoder_selfattentionlayer_2_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:	*
dtype0�
Omodel_6/transformer_encoder_19/Encoder-SelfAttentionLayer-2/query/einsum/EinsumEinsumGmodel_6/transformer_encoder_19/Encoder-1st-NormalizationLayer-2/add:z:0fmodel_6/transformer_encoder_19/Encoder-SelfAttentionLayer-2/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������P*
equationabc,cde->abde�
Tmodel_6/transformer_encoder_19/Encoder-SelfAttentionLayer-2/query/add/ReadVariableOpReadVariableOp]model_6_transformer_encoder_19_encoder_selfattentionlayer_2_query_add_readvariableop_resource*
_output_shapes

:*
dtype0�
Emodel_6/transformer_encoder_19/Encoder-SelfAttentionLayer-2/query/addAddV2Xmodel_6/transformer_encoder_19/Encoder-SelfAttentionLayer-2/query/einsum/Einsum:output:0\model_6/transformer_encoder_19/Encoder-SelfAttentionLayer-2/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������P�
\model_6/transformer_encoder_19/Encoder-SelfAttentionLayer-2/key/einsum/Einsum/ReadVariableOpReadVariableOpemodel_6_transformer_encoder_19_encoder_selfattentionlayer_2_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:	*
dtype0�
Mmodel_6/transformer_encoder_19/Encoder-SelfAttentionLayer-2/key/einsum/EinsumEinsumGmodel_6/transformer_encoder_19/Encoder-1st-NormalizationLayer-2/add:z:0dmodel_6/transformer_encoder_19/Encoder-SelfAttentionLayer-2/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������P*
equationabc,cde->abde�
Rmodel_6/transformer_encoder_19/Encoder-SelfAttentionLayer-2/key/add/ReadVariableOpReadVariableOp[model_6_transformer_encoder_19_encoder_selfattentionlayer_2_key_add_readvariableop_resource*
_output_shapes

:*
dtype0�
Cmodel_6/transformer_encoder_19/Encoder-SelfAttentionLayer-2/key/addAddV2Vmodel_6/transformer_encoder_19/Encoder-SelfAttentionLayer-2/key/einsum/Einsum:output:0Zmodel_6/transformer_encoder_19/Encoder-SelfAttentionLayer-2/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������P�
^model_6/transformer_encoder_19/Encoder-SelfAttentionLayer-2/value/einsum/Einsum/ReadVariableOpReadVariableOpgmodel_6_transformer_encoder_19_encoder_selfattentionlayer_2_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:	*
dtype0�
Omodel_6/transformer_encoder_19/Encoder-SelfAttentionLayer-2/value/einsum/EinsumEinsumGmodel_6/transformer_encoder_19/Encoder-1st-NormalizationLayer-2/add:z:0fmodel_6/transformer_encoder_19/Encoder-SelfAttentionLayer-2/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������P*
equationabc,cde->abde�
Tmodel_6/transformer_encoder_19/Encoder-SelfAttentionLayer-2/value/add/ReadVariableOpReadVariableOp]model_6_transformer_encoder_19_encoder_selfattentionlayer_2_value_add_readvariableop_resource*
_output_shapes

:*
dtype0�
Emodel_6/transformer_encoder_19/Encoder-SelfAttentionLayer-2/value/addAddV2Xmodel_6/transformer_encoder_19/Encoder-SelfAttentionLayer-2/value/einsum/Einsum:output:0\model_6/transformer_encoder_19/Encoder-SelfAttentionLayer-2/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������P�
Amodel_6/transformer_encoder_19/Encoder-SelfAttentionLayer-2/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *:�?�
?model_6/transformer_encoder_19/Encoder-SelfAttentionLayer-2/MulMulImodel_6/transformer_encoder_19/Encoder-SelfAttentionLayer-2/query/add:z:0Jmodel_6/transformer_encoder_19/Encoder-SelfAttentionLayer-2/Mul/y:output:0*
T0*/
_output_shapes
:���������P�
Imodel_6/transformer_encoder_19/Encoder-SelfAttentionLayer-2/einsum/EinsumEinsumGmodel_6/transformer_encoder_19/Encoder-SelfAttentionLayer-2/key/add:z:0Cmodel_6/transformer_encoder_19/Encoder-SelfAttentionLayer-2/Mul:z:0*
N*
T0*/
_output_shapes
:���������PP*
equationaecd,abcd->acbe�
Kmodel_6/transformer_encoder_19/Encoder-SelfAttentionLayer-2/softmax/SoftmaxSoftmaxRmodel_6/transformer_encoder_19/Encoder-SelfAttentionLayer-2/einsum/Einsum:output:0*
T0*/
_output_shapes
:���������PP�
Lmodel_6/transformer_encoder_19/Encoder-SelfAttentionLayer-2/dropout/IdentityIdentityUmodel_6/transformer_encoder_19/Encoder-SelfAttentionLayer-2/softmax/Softmax:softmax:0*
T0*/
_output_shapes
:���������PP�
Kmodel_6/transformer_encoder_19/Encoder-SelfAttentionLayer-2/einsum_1/EinsumEinsumUmodel_6/transformer_encoder_19/Encoder-SelfAttentionLayer-2/dropout/Identity:output:0Imodel_6/transformer_encoder_19/Encoder-SelfAttentionLayer-2/value/add:z:0*
N*
T0*/
_output_shapes
:���������P*
equationacbe,aecd->abcd�
imodel_6/transformer_encoder_19/Encoder-SelfAttentionLayer-2/attention_output/einsum/Einsum/ReadVariableOpReadVariableOprmodel_6_transformer_encoder_19_encoder_selfattentionlayer_2_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:	*
dtype0�
Zmodel_6/transformer_encoder_19/Encoder-SelfAttentionLayer-2/attention_output/einsum/EinsumEinsumTmodel_6/transformer_encoder_19/Encoder-SelfAttentionLayer-2/einsum_1/Einsum:output:0qmodel_6/transformer_encoder_19/Encoder-SelfAttentionLayer-2/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:���������P	*
equationabcd,cde->abe�
_model_6/transformer_encoder_19/Encoder-SelfAttentionLayer-2/attention_output/add/ReadVariableOpReadVariableOphmodel_6_transformer_encoder_19_encoder_selfattentionlayer_2_attention_output_add_readvariableop_resource*
_output_shapes
:	*
dtype0�
Pmodel_6/transformer_encoder_19/Encoder-SelfAttentionLayer-2/attention_output/addAddV2cmodel_6/transformer_encoder_19/Encoder-SelfAttentionLayer-2/attention_output/einsum/Einsum:output:0gmodel_6/transformer_encoder_19/Encoder-SelfAttentionLayer-2/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������P	�
>model_6/transformer_encoder_19/Encoder-1st-AdditionLayer-2/addAddV2Bmodel_6/transformer_encoder_18/Encoder-2nd-AdditionLayer-1/add:z:0Tmodel_6/transformer_encoder_19/Encoder-SelfAttentionLayer-2/attention_output/add:z:0*
T0*+
_output_shapes
:���������P	�
Emodel_6/transformer_encoder_19/Encoder-2nd-NormalizationLayer-2/ShapeShapeBmodel_6/transformer_encoder_19/Encoder-1st-AdditionLayer-2/add:z:0*
T0*
_output_shapes
::���
Smodel_6/transformer_encoder_19/Encoder-2nd-NormalizationLayer-2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
Umodel_6/transformer_encoder_19/Encoder-2nd-NormalizationLayer-2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
Umodel_6/transformer_encoder_19/Encoder-2nd-NormalizationLayer-2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
Mmodel_6/transformer_encoder_19/Encoder-2nd-NormalizationLayer-2/strided_sliceStridedSliceNmodel_6/transformer_encoder_19/Encoder-2nd-NormalizationLayer-2/Shape:output:0\model_6/transformer_encoder_19/Encoder-2nd-NormalizationLayer-2/strided_slice/stack:output:0^model_6/transformer_encoder_19/Encoder-2nd-NormalizationLayer-2/strided_slice/stack_1:output:0^model_6/transformer_encoder_19/Encoder-2nd-NormalizationLayer-2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask�
Emodel_6/transformer_encoder_19/Encoder-2nd-NormalizationLayer-2/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
Dmodel_6/transformer_encoder_19/Encoder-2nd-NormalizationLayer-2/ProdProdVmodel_6/transformer_encoder_19/Encoder-2nd-NormalizationLayer-2/strided_slice:output:0Nmodel_6/transformer_encoder_19/Encoder-2nd-NormalizationLayer-2/Const:output:0*
T0*
_output_shapes
: �
Umodel_6/transformer_encoder_19/Encoder-2nd-NormalizationLayer-2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:�
Wmodel_6/transformer_encoder_19/Encoder-2nd-NormalizationLayer-2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: �
Wmodel_6/transformer_encoder_19/Encoder-2nd-NormalizationLayer-2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
Omodel_6/transformer_encoder_19/Encoder-2nd-NormalizationLayer-2/strided_slice_1StridedSliceNmodel_6/transformer_encoder_19/Encoder-2nd-NormalizationLayer-2/Shape:output:0^model_6/transformer_encoder_19/Encoder-2nd-NormalizationLayer-2/strided_slice_1/stack:output:0`model_6/transformer_encoder_19/Encoder-2nd-NormalizationLayer-2/strided_slice_1/stack_1:output:0`model_6/transformer_encoder_19/Encoder-2nd-NormalizationLayer-2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask�
Gmodel_6/transformer_encoder_19/Encoder-2nd-NormalizationLayer-2/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
Fmodel_6/transformer_encoder_19/Encoder-2nd-NormalizationLayer-2/Prod_1ProdXmodel_6/transformer_encoder_19/Encoder-2nd-NormalizationLayer-2/strided_slice_1:output:0Pmodel_6/transformer_encoder_19/Encoder-2nd-NormalizationLayer-2/Const_1:output:0*
T0*
_output_shapes
: �
Omodel_6/transformer_encoder_19/Encoder-2nd-NormalizationLayer-2/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :�
Omodel_6/transformer_encoder_19/Encoder-2nd-NormalizationLayer-2/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :�
Mmodel_6/transformer_encoder_19/Encoder-2nd-NormalizationLayer-2/Reshape/shapePackXmodel_6/transformer_encoder_19/Encoder-2nd-NormalizationLayer-2/Reshape/shape/0:output:0Mmodel_6/transformer_encoder_19/Encoder-2nd-NormalizationLayer-2/Prod:output:0Omodel_6/transformer_encoder_19/Encoder-2nd-NormalizationLayer-2/Prod_1:output:0Xmodel_6/transformer_encoder_19/Encoder-2nd-NormalizationLayer-2/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:�
Gmodel_6/transformer_encoder_19/Encoder-2nd-NormalizationLayer-2/ReshapeReshapeBmodel_6/transformer_encoder_19/Encoder-1st-AdditionLayer-2/add:z:0Vmodel_6/transformer_encoder_19/Encoder-2nd-NormalizationLayer-2/Reshape/shape:output:0*
T0*8
_output_shapes&
$:"�������������������
Kmodel_6/transformer_encoder_19/Encoder-2nd-NormalizationLayer-2/ones/packedPackMmodel_6/transformer_encoder_19/Encoder-2nd-NormalizationLayer-2/Prod:output:0*
N*
T0*
_output_shapes
:�
Jmodel_6/transformer_encoder_19/Encoder-2nd-NormalizationLayer-2/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
Dmodel_6/transformer_encoder_19/Encoder-2nd-NormalizationLayer-2/onesFillTmodel_6/transformer_encoder_19/Encoder-2nd-NormalizationLayer-2/ones/packed:output:0Smodel_6/transformer_encoder_19/Encoder-2nd-NormalizationLayer-2/ones/Const:output:0*
T0*#
_output_shapes
:����������
Lmodel_6/transformer_encoder_19/Encoder-2nd-NormalizationLayer-2/zeros/packedPackMmodel_6/transformer_encoder_19/Encoder-2nd-NormalizationLayer-2/Prod:output:0*
N*
T0*
_output_shapes
:�
Kmodel_6/transformer_encoder_19/Encoder-2nd-NormalizationLayer-2/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
Emodel_6/transformer_encoder_19/Encoder-2nd-NormalizationLayer-2/zerosFillUmodel_6/transformer_encoder_19/Encoder-2nd-NormalizationLayer-2/zeros/packed:output:0Tmodel_6/transformer_encoder_19/Encoder-2nd-NormalizationLayer-2/zeros/Const:output:0*
T0*#
_output_shapes
:����������
Gmodel_6/transformer_encoder_19/Encoder-2nd-NormalizationLayer-2/Const_2Const*
_output_shapes
: *
dtype0*
valueB �
Gmodel_6/transformer_encoder_19/Encoder-2nd-NormalizationLayer-2/Const_3Const*
_output_shapes
: *
dtype0*
valueB �
Pmodel_6/transformer_encoder_19/Encoder-2nd-NormalizationLayer-2/FusedBatchNormV3FusedBatchNormV3Pmodel_6/transformer_encoder_19/Encoder-2nd-NormalizationLayer-2/Reshape:output:0Mmodel_6/transformer_encoder_19/Encoder-2nd-NormalizationLayer-2/ones:output:0Nmodel_6/transformer_encoder_19/Encoder-2nd-NormalizationLayer-2/zeros:output:0Pmodel_6/transformer_encoder_19/Encoder-2nd-NormalizationLayer-2/Const_2:output:0Pmodel_6/transformer_encoder_19/Encoder-2nd-NormalizationLayer-2/Const_3:output:0*
T0*
U0*x
_output_shapesf
d:"������������������:���������:���������:���������:���������:*
data_formatNCHW*
epsilon%o�:�
Imodel_6/transformer_encoder_19/Encoder-2nd-NormalizationLayer-2/Reshape_1ReshapeTmodel_6/transformer_encoder_19/Encoder-2nd-NormalizationLayer-2/FusedBatchNormV3:y:0Nmodel_6/transformer_encoder_19/Encoder-2nd-NormalizationLayer-2/Shape:output:0*
T0*+
_output_shapes
:���������P	�
Rmodel_6/transformer_encoder_19/Encoder-2nd-NormalizationLayer-2/mul/ReadVariableOpReadVariableOp[model_6_transformer_encoder_19_encoder_2nd_normalizationlayer_2_mul_readvariableop_resource*
_output_shapes
:	*
dtype0�
Cmodel_6/transformer_encoder_19/Encoder-2nd-NormalizationLayer-2/mulMulRmodel_6/transformer_encoder_19/Encoder-2nd-NormalizationLayer-2/Reshape_1:output:0Zmodel_6/transformer_encoder_19/Encoder-2nd-NormalizationLayer-2/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������P	�
Rmodel_6/transformer_encoder_19/Encoder-2nd-NormalizationLayer-2/add/ReadVariableOpReadVariableOp[model_6_transformer_encoder_19_encoder_2nd_normalizationlayer_2_add_readvariableop_resource*
_output_shapes
:	*
dtype0�
Cmodel_6/transformer_encoder_19/Encoder-2nd-NormalizationLayer-2/addAddV2Gmodel_6/transformer_encoder_19/Encoder-2nd-NormalizationLayer-2/mul:z:0Zmodel_6/transformer_encoder_19/Encoder-2nd-NormalizationLayer-2/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������P	�
Tmodel_6/transformer_encoder_19/Encoder-FeedForwardLayer_1_2/Tensordot/ReadVariableOpReadVariableOp]model_6_transformer_encoder_19_encoder_feedforwardlayer_1_2_tensordot_readvariableop_resource*
_output_shapes

:		*
dtype0�
Jmodel_6/transformer_encoder_19/Encoder-FeedForwardLayer_1_2/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:�
Jmodel_6/transformer_encoder_19/Encoder-FeedForwardLayer_1_2/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       �
Kmodel_6/transformer_encoder_19/Encoder-FeedForwardLayer_1_2/Tensordot/ShapeShapeGmodel_6/transformer_encoder_19/Encoder-2nd-NormalizationLayer-2/add:z:0*
T0*
_output_shapes
::���
Smodel_6/transformer_encoder_19/Encoder-FeedForwardLayer_1_2/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Nmodel_6/transformer_encoder_19/Encoder-FeedForwardLayer_1_2/Tensordot/GatherV2GatherV2Tmodel_6/transformer_encoder_19/Encoder-FeedForwardLayer_1_2/Tensordot/Shape:output:0Smodel_6/transformer_encoder_19/Encoder-FeedForwardLayer_1_2/Tensordot/free:output:0\model_6/transformer_encoder_19/Encoder-FeedForwardLayer_1_2/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
Umodel_6/transformer_encoder_19/Encoder-FeedForwardLayer_1_2/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Pmodel_6/transformer_encoder_19/Encoder-FeedForwardLayer_1_2/Tensordot/GatherV2_1GatherV2Tmodel_6/transformer_encoder_19/Encoder-FeedForwardLayer_1_2/Tensordot/Shape:output:0Smodel_6/transformer_encoder_19/Encoder-FeedForwardLayer_1_2/Tensordot/axes:output:0^model_6/transformer_encoder_19/Encoder-FeedForwardLayer_1_2/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
Kmodel_6/transformer_encoder_19/Encoder-FeedForwardLayer_1_2/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
Jmodel_6/transformer_encoder_19/Encoder-FeedForwardLayer_1_2/Tensordot/ProdProdWmodel_6/transformer_encoder_19/Encoder-FeedForwardLayer_1_2/Tensordot/GatherV2:output:0Tmodel_6/transformer_encoder_19/Encoder-FeedForwardLayer_1_2/Tensordot/Const:output:0*
T0*
_output_shapes
: �
Mmodel_6/transformer_encoder_19/Encoder-FeedForwardLayer_1_2/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
Lmodel_6/transformer_encoder_19/Encoder-FeedForwardLayer_1_2/Tensordot/Prod_1ProdYmodel_6/transformer_encoder_19/Encoder-FeedForwardLayer_1_2/Tensordot/GatherV2_1:output:0Vmodel_6/transformer_encoder_19/Encoder-FeedForwardLayer_1_2/Tensordot/Const_1:output:0*
T0*
_output_shapes
: �
Qmodel_6/transformer_encoder_19/Encoder-FeedForwardLayer_1_2/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Lmodel_6/transformer_encoder_19/Encoder-FeedForwardLayer_1_2/Tensordot/concatConcatV2Smodel_6/transformer_encoder_19/Encoder-FeedForwardLayer_1_2/Tensordot/free:output:0Smodel_6/transformer_encoder_19/Encoder-FeedForwardLayer_1_2/Tensordot/axes:output:0Zmodel_6/transformer_encoder_19/Encoder-FeedForwardLayer_1_2/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
Kmodel_6/transformer_encoder_19/Encoder-FeedForwardLayer_1_2/Tensordot/stackPackSmodel_6/transformer_encoder_19/Encoder-FeedForwardLayer_1_2/Tensordot/Prod:output:0Umodel_6/transformer_encoder_19/Encoder-FeedForwardLayer_1_2/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
Omodel_6/transformer_encoder_19/Encoder-FeedForwardLayer_1_2/Tensordot/transpose	TransposeGmodel_6/transformer_encoder_19/Encoder-2nd-NormalizationLayer-2/add:z:0Umodel_6/transformer_encoder_19/Encoder-FeedForwardLayer_1_2/Tensordot/concat:output:0*
T0*+
_output_shapes
:���������P	�
Mmodel_6/transformer_encoder_19/Encoder-FeedForwardLayer_1_2/Tensordot/ReshapeReshapeSmodel_6/transformer_encoder_19/Encoder-FeedForwardLayer_1_2/Tensordot/transpose:y:0Tmodel_6/transformer_encoder_19/Encoder-FeedForwardLayer_1_2/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
Lmodel_6/transformer_encoder_19/Encoder-FeedForwardLayer_1_2/Tensordot/MatMulMatMulVmodel_6/transformer_encoder_19/Encoder-FeedForwardLayer_1_2/Tensordot/Reshape:output:0\model_6/transformer_encoder_19/Encoder-FeedForwardLayer_1_2/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������	�
Mmodel_6/transformer_encoder_19/Encoder-FeedForwardLayer_1_2/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:	�
Smodel_6/transformer_encoder_19/Encoder-FeedForwardLayer_1_2/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Nmodel_6/transformer_encoder_19/Encoder-FeedForwardLayer_1_2/Tensordot/concat_1ConcatV2Wmodel_6/transformer_encoder_19/Encoder-FeedForwardLayer_1_2/Tensordot/GatherV2:output:0Vmodel_6/transformer_encoder_19/Encoder-FeedForwardLayer_1_2/Tensordot/Const_2:output:0\model_6/transformer_encoder_19/Encoder-FeedForwardLayer_1_2/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
Emodel_6/transformer_encoder_19/Encoder-FeedForwardLayer_1_2/TensordotReshapeVmodel_6/transformer_encoder_19/Encoder-FeedForwardLayer_1_2/Tensordot/MatMul:product:0Wmodel_6/transformer_encoder_19/Encoder-FeedForwardLayer_1_2/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������P	�
Rmodel_6/transformer_encoder_19/Encoder-FeedForwardLayer_1_2/BiasAdd/ReadVariableOpReadVariableOp[model_6_transformer_encoder_19_encoder_feedforwardlayer_1_2_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype0�
Cmodel_6/transformer_encoder_19/Encoder-FeedForwardLayer_1_2/BiasAddBiasAddNmodel_6/transformer_encoder_19/Encoder-FeedForwardLayer_1_2/Tensordot:output:0Zmodel_6/transformer_encoder_19/Encoder-FeedForwardLayer_1_2/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������P	�
Fmodel_6/transformer_encoder_19/Encoder-FeedForwardLayer_1_2/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
Dmodel_6/transformer_encoder_19/Encoder-FeedForwardLayer_1_2/Gelu/mulMulOmodel_6/transformer_encoder_19/Encoder-FeedForwardLayer_1_2/Gelu/mul/x:output:0Lmodel_6/transformer_encoder_19/Encoder-FeedForwardLayer_1_2/BiasAdd:output:0*
T0*+
_output_shapes
:���������P	�
Gmodel_6/transformer_encoder_19/Encoder-FeedForwardLayer_1_2/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *��?�
Hmodel_6/transformer_encoder_19/Encoder-FeedForwardLayer_1_2/Gelu/truedivRealDivLmodel_6/transformer_encoder_19/Encoder-FeedForwardLayer_1_2/BiasAdd:output:0Pmodel_6/transformer_encoder_19/Encoder-FeedForwardLayer_1_2/Gelu/Cast/x:output:0*
T0*+
_output_shapes
:���������P	�
Dmodel_6/transformer_encoder_19/Encoder-FeedForwardLayer_1_2/Gelu/ErfErfLmodel_6/transformer_encoder_19/Encoder-FeedForwardLayer_1_2/Gelu/truediv:z:0*
T0*+
_output_shapes
:���������P	�
Fmodel_6/transformer_encoder_19/Encoder-FeedForwardLayer_1_2/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
Dmodel_6/transformer_encoder_19/Encoder-FeedForwardLayer_1_2/Gelu/addAddV2Omodel_6/transformer_encoder_19/Encoder-FeedForwardLayer_1_2/Gelu/add/x:output:0Hmodel_6/transformer_encoder_19/Encoder-FeedForwardLayer_1_2/Gelu/Erf:y:0*
T0*+
_output_shapes
:���������P	�
Fmodel_6/transformer_encoder_19/Encoder-FeedForwardLayer_1_2/Gelu/mul_1MulHmodel_6/transformer_encoder_19/Encoder-FeedForwardLayer_1_2/Gelu/mul:z:0Hmodel_6/transformer_encoder_19/Encoder-FeedForwardLayer_1_2/Gelu/add:z:0*
T0*+
_output_shapes
:���������P	�
Tmodel_6/transformer_encoder_19/Encoder-FeedForwardLayer_2_2/Tensordot/ReadVariableOpReadVariableOp]model_6_transformer_encoder_19_encoder_feedforwardlayer_2_2_tensordot_readvariableop_resource*
_output_shapes

:		*
dtype0�
Jmodel_6/transformer_encoder_19/Encoder-FeedForwardLayer_2_2/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:�
Jmodel_6/transformer_encoder_19/Encoder-FeedForwardLayer_2_2/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       �
Kmodel_6/transformer_encoder_19/Encoder-FeedForwardLayer_2_2/Tensordot/ShapeShapeJmodel_6/transformer_encoder_19/Encoder-FeedForwardLayer_1_2/Gelu/mul_1:z:0*
T0*
_output_shapes
::���
Smodel_6/transformer_encoder_19/Encoder-FeedForwardLayer_2_2/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Nmodel_6/transformer_encoder_19/Encoder-FeedForwardLayer_2_2/Tensordot/GatherV2GatherV2Tmodel_6/transformer_encoder_19/Encoder-FeedForwardLayer_2_2/Tensordot/Shape:output:0Smodel_6/transformer_encoder_19/Encoder-FeedForwardLayer_2_2/Tensordot/free:output:0\model_6/transformer_encoder_19/Encoder-FeedForwardLayer_2_2/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
Umodel_6/transformer_encoder_19/Encoder-FeedForwardLayer_2_2/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Pmodel_6/transformer_encoder_19/Encoder-FeedForwardLayer_2_2/Tensordot/GatherV2_1GatherV2Tmodel_6/transformer_encoder_19/Encoder-FeedForwardLayer_2_2/Tensordot/Shape:output:0Smodel_6/transformer_encoder_19/Encoder-FeedForwardLayer_2_2/Tensordot/axes:output:0^model_6/transformer_encoder_19/Encoder-FeedForwardLayer_2_2/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
Kmodel_6/transformer_encoder_19/Encoder-FeedForwardLayer_2_2/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
Jmodel_6/transformer_encoder_19/Encoder-FeedForwardLayer_2_2/Tensordot/ProdProdWmodel_6/transformer_encoder_19/Encoder-FeedForwardLayer_2_2/Tensordot/GatherV2:output:0Tmodel_6/transformer_encoder_19/Encoder-FeedForwardLayer_2_2/Tensordot/Const:output:0*
T0*
_output_shapes
: �
Mmodel_6/transformer_encoder_19/Encoder-FeedForwardLayer_2_2/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
Lmodel_6/transformer_encoder_19/Encoder-FeedForwardLayer_2_2/Tensordot/Prod_1ProdYmodel_6/transformer_encoder_19/Encoder-FeedForwardLayer_2_2/Tensordot/GatherV2_1:output:0Vmodel_6/transformer_encoder_19/Encoder-FeedForwardLayer_2_2/Tensordot/Const_1:output:0*
T0*
_output_shapes
: �
Qmodel_6/transformer_encoder_19/Encoder-FeedForwardLayer_2_2/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Lmodel_6/transformer_encoder_19/Encoder-FeedForwardLayer_2_2/Tensordot/concatConcatV2Smodel_6/transformer_encoder_19/Encoder-FeedForwardLayer_2_2/Tensordot/free:output:0Smodel_6/transformer_encoder_19/Encoder-FeedForwardLayer_2_2/Tensordot/axes:output:0Zmodel_6/transformer_encoder_19/Encoder-FeedForwardLayer_2_2/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
Kmodel_6/transformer_encoder_19/Encoder-FeedForwardLayer_2_2/Tensordot/stackPackSmodel_6/transformer_encoder_19/Encoder-FeedForwardLayer_2_2/Tensordot/Prod:output:0Umodel_6/transformer_encoder_19/Encoder-FeedForwardLayer_2_2/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
Omodel_6/transformer_encoder_19/Encoder-FeedForwardLayer_2_2/Tensordot/transpose	TransposeJmodel_6/transformer_encoder_19/Encoder-FeedForwardLayer_1_2/Gelu/mul_1:z:0Umodel_6/transformer_encoder_19/Encoder-FeedForwardLayer_2_2/Tensordot/concat:output:0*
T0*+
_output_shapes
:���������P	�
Mmodel_6/transformer_encoder_19/Encoder-FeedForwardLayer_2_2/Tensordot/ReshapeReshapeSmodel_6/transformer_encoder_19/Encoder-FeedForwardLayer_2_2/Tensordot/transpose:y:0Tmodel_6/transformer_encoder_19/Encoder-FeedForwardLayer_2_2/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
Lmodel_6/transformer_encoder_19/Encoder-FeedForwardLayer_2_2/Tensordot/MatMulMatMulVmodel_6/transformer_encoder_19/Encoder-FeedForwardLayer_2_2/Tensordot/Reshape:output:0\model_6/transformer_encoder_19/Encoder-FeedForwardLayer_2_2/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������	�
Mmodel_6/transformer_encoder_19/Encoder-FeedForwardLayer_2_2/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:	�
Smodel_6/transformer_encoder_19/Encoder-FeedForwardLayer_2_2/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Nmodel_6/transformer_encoder_19/Encoder-FeedForwardLayer_2_2/Tensordot/concat_1ConcatV2Wmodel_6/transformer_encoder_19/Encoder-FeedForwardLayer_2_2/Tensordot/GatherV2:output:0Vmodel_6/transformer_encoder_19/Encoder-FeedForwardLayer_2_2/Tensordot/Const_2:output:0\model_6/transformer_encoder_19/Encoder-FeedForwardLayer_2_2/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
Emodel_6/transformer_encoder_19/Encoder-FeedForwardLayer_2_2/TensordotReshapeVmodel_6/transformer_encoder_19/Encoder-FeedForwardLayer_2_2/Tensordot/MatMul:product:0Wmodel_6/transformer_encoder_19/Encoder-FeedForwardLayer_2_2/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������P	�
Rmodel_6/transformer_encoder_19/Encoder-FeedForwardLayer_2_2/BiasAdd/ReadVariableOpReadVariableOp[model_6_transformer_encoder_19_encoder_feedforwardlayer_2_2_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype0�
Cmodel_6/transformer_encoder_19/Encoder-FeedForwardLayer_2_2/BiasAddBiasAddNmodel_6/transformer_encoder_19/Encoder-FeedForwardLayer_2_2/Tensordot:output:0Zmodel_6/transformer_encoder_19/Encoder-FeedForwardLayer_2_2/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������P	�
2model_6/transformer_encoder_19/dropout_19/IdentityIdentityLmodel_6/transformer_encoder_19/Encoder-FeedForwardLayer_2_2/BiasAdd:output:0*
T0*+
_output_shapes
:���������P	�
>model_6/transformer_encoder_19/Encoder-2nd-AdditionLayer-2/addAddV2Bmodel_6/transformer_encoder_19/Encoder-1st-AdditionLayer-2/add:z:0;model_6/transformer_encoder_19/dropout_19/Identity:output:0*
T0*+
_output_shapes
:���������P	�
Emodel_6/transformer_encoder_20/Encoder-1st-NormalizationLayer-3/ShapeShapeBmodel_6/transformer_encoder_19/Encoder-2nd-AdditionLayer-2/add:z:0*
T0*
_output_shapes
::���
Smodel_6/transformer_encoder_20/Encoder-1st-NormalizationLayer-3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
Umodel_6/transformer_encoder_20/Encoder-1st-NormalizationLayer-3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
Umodel_6/transformer_encoder_20/Encoder-1st-NormalizationLayer-3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
Mmodel_6/transformer_encoder_20/Encoder-1st-NormalizationLayer-3/strided_sliceStridedSliceNmodel_6/transformer_encoder_20/Encoder-1st-NormalizationLayer-3/Shape:output:0\model_6/transformer_encoder_20/Encoder-1st-NormalizationLayer-3/strided_slice/stack:output:0^model_6/transformer_encoder_20/Encoder-1st-NormalizationLayer-3/strided_slice/stack_1:output:0^model_6/transformer_encoder_20/Encoder-1st-NormalizationLayer-3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask�
Emodel_6/transformer_encoder_20/Encoder-1st-NormalizationLayer-3/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
Dmodel_6/transformer_encoder_20/Encoder-1st-NormalizationLayer-3/ProdProdVmodel_6/transformer_encoder_20/Encoder-1st-NormalizationLayer-3/strided_slice:output:0Nmodel_6/transformer_encoder_20/Encoder-1st-NormalizationLayer-3/Const:output:0*
T0*
_output_shapes
: �
Umodel_6/transformer_encoder_20/Encoder-1st-NormalizationLayer-3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:�
Wmodel_6/transformer_encoder_20/Encoder-1st-NormalizationLayer-3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: �
Wmodel_6/transformer_encoder_20/Encoder-1st-NormalizationLayer-3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
Omodel_6/transformer_encoder_20/Encoder-1st-NormalizationLayer-3/strided_slice_1StridedSliceNmodel_6/transformer_encoder_20/Encoder-1st-NormalizationLayer-3/Shape:output:0^model_6/transformer_encoder_20/Encoder-1st-NormalizationLayer-3/strided_slice_1/stack:output:0`model_6/transformer_encoder_20/Encoder-1st-NormalizationLayer-3/strided_slice_1/stack_1:output:0`model_6/transformer_encoder_20/Encoder-1st-NormalizationLayer-3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask�
Gmodel_6/transformer_encoder_20/Encoder-1st-NormalizationLayer-3/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
Fmodel_6/transformer_encoder_20/Encoder-1st-NormalizationLayer-3/Prod_1ProdXmodel_6/transformer_encoder_20/Encoder-1st-NormalizationLayer-3/strided_slice_1:output:0Pmodel_6/transformer_encoder_20/Encoder-1st-NormalizationLayer-3/Const_1:output:0*
T0*
_output_shapes
: �
Omodel_6/transformer_encoder_20/Encoder-1st-NormalizationLayer-3/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :�
Omodel_6/transformer_encoder_20/Encoder-1st-NormalizationLayer-3/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :�
Mmodel_6/transformer_encoder_20/Encoder-1st-NormalizationLayer-3/Reshape/shapePackXmodel_6/transformer_encoder_20/Encoder-1st-NormalizationLayer-3/Reshape/shape/0:output:0Mmodel_6/transformer_encoder_20/Encoder-1st-NormalizationLayer-3/Prod:output:0Omodel_6/transformer_encoder_20/Encoder-1st-NormalizationLayer-3/Prod_1:output:0Xmodel_6/transformer_encoder_20/Encoder-1st-NormalizationLayer-3/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:�
Gmodel_6/transformer_encoder_20/Encoder-1st-NormalizationLayer-3/ReshapeReshapeBmodel_6/transformer_encoder_19/Encoder-2nd-AdditionLayer-2/add:z:0Vmodel_6/transformer_encoder_20/Encoder-1st-NormalizationLayer-3/Reshape/shape:output:0*
T0*8
_output_shapes&
$:"�������������������
Kmodel_6/transformer_encoder_20/Encoder-1st-NormalizationLayer-3/ones/packedPackMmodel_6/transformer_encoder_20/Encoder-1st-NormalizationLayer-3/Prod:output:0*
N*
T0*
_output_shapes
:�
Jmodel_6/transformer_encoder_20/Encoder-1st-NormalizationLayer-3/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
Dmodel_6/transformer_encoder_20/Encoder-1st-NormalizationLayer-3/onesFillTmodel_6/transformer_encoder_20/Encoder-1st-NormalizationLayer-3/ones/packed:output:0Smodel_6/transformer_encoder_20/Encoder-1st-NormalizationLayer-3/ones/Const:output:0*
T0*#
_output_shapes
:����������
Lmodel_6/transformer_encoder_20/Encoder-1st-NormalizationLayer-3/zeros/packedPackMmodel_6/transformer_encoder_20/Encoder-1st-NormalizationLayer-3/Prod:output:0*
N*
T0*
_output_shapes
:�
Kmodel_6/transformer_encoder_20/Encoder-1st-NormalizationLayer-3/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
Emodel_6/transformer_encoder_20/Encoder-1st-NormalizationLayer-3/zerosFillUmodel_6/transformer_encoder_20/Encoder-1st-NormalizationLayer-3/zeros/packed:output:0Tmodel_6/transformer_encoder_20/Encoder-1st-NormalizationLayer-3/zeros/Const:output:0*
T0*#
_output_shapes
:����������
Gmodel_6/transformer_encoder_20/Encoder-1st-NormalizationLayer-3/Const_2Const*
_output_shapes
: *
dtype0*
valueB �
Gmodel_6/transformer_encoder_20/Encoder-1st-NormalizationLayer-3/Const_3Const*
_output_shapes
: *
dtype0*
valueB �
Pmodel_6/transformer_encoder_20/Encoder-1st-NormalizationLayer-3/FusedBatchNormV3FusedBatchNormV3Pmodel_6/transformer_encoder_20/Encoder-1st-NormalizationLayer-3/Reshape:output:0Mmodel_6/transformer_encoder_20/Encoder-1st-NormalizationLayer-3/ones:output:0Nmodel_6/transformer_encoder_20/Encoder-1st-NormalizationLayer-3/zeros:output:0Pmodel_6/transformer_encoder_20/Encoder-1st-NormalizationLayer-3/Const_2:output:0Pmodel_6/transformer_encoder_20/Encoder-1st-NormalizationLayer-3/Const_3:output:0*
T0*
U0*x
_output_shapesf
d:"������������������:���������:���������:���������:���������:*
data_formatNCHW*
epsilon%o�:�
Imodel_6/transformer_encoder_20/Encoder-1st-NormalizationLayer-3/Reshape_1ReshapeTmodel_6/transformer_encoder_20/Encoder-1st-NormalizationLayer-3/FusedBatchNormV3:y:0Nmodel_6/transformer_encoder_20/Encoder-1st-NormalizationLayer-3/Shape:output:0*
T0*+
_output_shapes
:���������P	�
Rmodel_6/transformer_encoder_20/Encoder-1st-NormalizationLayer-3/mul/ReadVariableOpReadVariableOp[model_6_transformer_encoder_20_encoder_1st_normalizationlayer_3_mul_readvariableop_resource*
_output_shapes
:	*
dtype0�
Cmodel_6/transformer_encoder_20/Encoder-1st-NormalizationLayer-3/mulMulRmodel_6/transformer_encoder_20/Encoder-1st-NormalizationLayer-3/Reshape_1:output:0Zmodel_6/transformer_encoder_20/Encoder-1st-NormalizationLayer-3/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������P	�
Rmodel_6/transformer_encoder_20/Encoder-1st-NormalizationLayer-3/add/ReadVariableOpReadVariableOp[model_6_transformer_encoder_20_encoder_1st_normalizationlayer_3_add_readvariableop_resource*
_output_shapes
:	*
dtype0�
Cmodel_6/transformer_encoder_20/Encoder-1st-NormalizationLayer-3/addAddV2Gmodel_6/transformer_encoder_20/Encoder-1st-NormalizationLayer-3/mul:z:0Zmodel_6/transformer_encoder_20/Encoder-1st-NormalizationLayer-3/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������P	�
^model_6/transformer_encoder_20/Encoder-SelfAttentionLayer-3/query/einsum/Einsum/ReadVariableOpReadVariableOpgmodel_6_transformer_encoder_20_encoder_selfattentionlayer_3_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:	*
dtype0�
Omodel_6/transformer_encoder_20/Encoder-SelfAttentionLayer-3/query/einsum/EinsumEinsumGmodel_6/transformer_encoder_20/Encoder-1st-NormalizationLayer-3/add:z:0fmodel_6/transformer_encoder_20/Encoder-SelfAttentionLayer-3/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������P*
equationabc,cde->abde�
Tmodel_6/transformer_encoder_20/Encoder-SelfAttentionLayer-3/query/add/ReadVariableOpReadVariableOp]model_6_transformer_encoder_20_encoder_selfattentionlayer_3_query_add_readvariableop_resource*
_output_shapes

:*
dtype0�
Emodel_6/transformer_encoder_20/Encoder-SelfAttentionLayer-3/query/addAddV2Xmodel_6/transformer_encoder_20/Encoder-SelfAttentionLayer-3/query/einsum/Einsum:output:0\model_6/transformer_encoder_20/Encoder-SelfAttentionLayer-3/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������P�
\model_6/transformer_encoder_20/Encoder-SelfAttentionLayer-3/key/einsum/Einsum/ReadVariableOpReadVariableOpemodel_6_transformer_encoder_20_encoder_selfattentionlayer_3_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:	*
dtype0�
Mmodel_6/transformer_encoder_20/Encoder-SelfAttentionLayer-3/key/einsum/EinsumEinsumGmodel_6/transformer_encoder_20/Encoder-1st-NormalizationLayer-3/add:z:0dmodel_6/transformer_encoder_20/Encoder-SelfAttentionLayer-3/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������P*
equationabc,cde->abde�
Rmodel_6/transformer_encoder_20/Encoder-SelfAttentionLayer-3/key/add/ReadVariableOpReadVariableOp[model_6_transformer_encoder_20_encoder_selfattentionlayer_3_key_add_readvariableop_resource*
_output_shapes

:*
dtype0�
Cmodel_6/transformer_encoder_20/Encoder-SelfAttentionLayer-3/key/addAddV2Vmodel_6/transformer_encoder_20/Encoder-SelfAttentionLayer-3/key/einsum/Einsum:output:0Zmodel_6/transformer_encoder_20/Encoder-SelfAttentionLayer-3/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������P�
^model_6/transformer_encoder_20/Encoder-SelfAttentionLayer-3/value/einsum/Einsum/ReadVariableOpReadVariableOpgmodel_6_transformer_encoder_20_encoder_selfattentionlayer_3_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:	*
dtype0�
Omodel_6/transformer_encoder_20/Encoder-SelfAttentionLayer-3/value/einsum/EinsumEinsumGmodel_6/transformer_encoder_20/Encoder-1st-NormalizationLayer-3/add:z:0fmodel_6/transformer_encoder_20/Encoder-SelfAttentionLayer-3/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������P*
equationabc,cde->abde�
Tmodel_6/transformer_encoder_20/Encoder-SelfAttentionLayer-3/value/add/ReadVariableOpReadVariableOp]model_6_transformer_encoder_20_encoder_selfattentionlayer_3_value_add_readvariableop_resource*
_output_shapes

:*
dtype0�
Emodel_6/transformer_encoder_20/Encoder-SelfAttentionLayer-3/value/addAddV2Xmodel_6/transformer_encoder_20/Encoder-SelfAttentionLayer-3/value/einsum/Einsum:output:0\model_6/transformer_encoder_20/Encoder-SelfAttentionLayer-3/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������P�
Amodel_6/transformer_encoder_20/Encoder-SelfAttentionLayer-3/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *:�?�
?model_6/transformer_encoder_20/Encoder-SelfAttentionLayer-3/MulMulImodel_6/transformer_encoder_20/Encoder-SelfAttentionLayer-3/query/add:z:0Jmodel_6/transformer_encoder_20/Encoder-SelfAttentionLayer-3/Mul/y:output:0*
T0*/
_output_shapes
:���������P�
Imodel_6/transformer_encoder_20/Encoder-SelfAttentionLayer-3/einsum/EinsumEinsumGmodel_6/transformer_encoder_20/Encoder-SelfAttentionLayer-3/key/add:z:0Cmodel_6/transformer_encoder_20/Encoder-SelfAttentionLayer-3/Mul:z:0*
N*
T0*/
_output_shapes
:���������PP*
equationaecd,abcd->acbe�
Kmodel_6/transformer_encoder_20/Encoder-SelfAttentionLayer-3/softmax/SoftmaxSoftmaxRmodel_6/transformer_encoder_20/Encoder-SelfAttentionLayer-3/einsum/Einsum:output:0*
T0*/
_output_shapes
:���������PP�
Lmodel_6/transformer_encoder_20/Encoder-SelfAttentionLayer-3/dropout/IdentityIdentityUmodel_6/transformer_encoder_20/Encoder-SelfAttentionLayer-3/softmax/Softmax:softmax:0*
T0*/
_output_shapes
:���������PP�
Kmodel_6/transformer_encoder_20/Encoder-SelfAttentionLayer-3/einsum_1/EinsumEinsumUmodel_6/transformer_encoder_20/Encoder-SelfAttentionLayer-3/dropout/Identity:output:0Imodel_6/transformer_encoder_20/Encoder-SelfAttentionLayer-3/value/add:z:0*
N*
T0*/
_output_shapes
:���������P*
equationacbe,aecd->abcd�
imodel_6/transformer_encoder_20/Encoder-SelfAttentionLayer-3/attention_output/einsum/Einsum/ReadVariableOpReadVariableOprmodel_6_transformer_encoder_20_encoder_selfattentionlayer_3_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:	*
dtype0�
Zmodel_6/transformer_encoder_20/Encoder-SelfAttentionLayer-3/attention_output/einsum/EinsumEinsumTmodel_6/transformer_encoder_20/Encoder-SelfAttentionLayer-3/einsum_1/Einsum:output:0qmodel_6/transformer_encoder_20/Encoder-SelfAttentionLayer-3/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:���������P	*
equationabcd,cde->abe�
_model_6/transformer_encoder_20/Encoder-SelfAttentionLayer-3/attention_output/add/ReadVariableOpReadVariableOphmodel_6_transformer_encoder_20_encoder_selfattentionlayer_3_attention_output_add_readvariableop_resource*
_output_shapes
:	*
dtype0�
Pmodel_6/transformer_encoder_20/Encoder-SelfAttentionLayer-3/attention_output/addAddV2cmodel_6/transformer_encoder_20/Encoder-SelfAttentionLayer-3/attention_output/einsum/Einsum:output:0gmodel_6/transformer_encoder_20/Encoder-SelfAttentionLayer-3/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������P	�
>model_6/transformer_encoder_20/Encoder-1st-AdditionLayer-3/addAddV2Bmodel_6/transformer_encoder_19/Encoder-2nd-AdditionLayer-2/add:z:0Tmodel_6/transformer_encoder_20/Encoder-SelfAttentionLayer-3/attention_output/add:z:0*
T0*+
_output_shapes
:���������P	�
Emodel_6/transformer_encoder_20/Encoder-2nd-NormalizationLayer-3/ShapeShapeBmodel_6/transformer_encoder_20/Encoder-1st-AdditionLayer-3/add:z:0*
T0*
_output_shapes
::���
Smodel_6/transformer_encoder_20/Encoder-2nd-NormalizationLayer-3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
Umodel_6/transformer_encoder_20/Encoder-2nd-NormalizationLayer-3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
Umodel_6/transformer_encoder_20/Encoder-2nd-NormalizationLayer-3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
Mmodel_6/transformer_encoder_20/Encoder-2nd-NormalizationLayer-3/strided_sliceStridedSliceNmodel_6/transformer_encoder_20/Encoder-2nd-NormalizationLayer-3/Shape:output:0\model_6/transformer_encoder_20/Encoder-2nd-NormalizationLayer-3/strided_slice/stack:output:0^model_6/transformer_encoder_20/Encoder-2nd-NormalizationLayer-3/strided_slice/stack_1:output:0^model_6/transformer_encoder_20/Encoder-2nd-NormalizationLayer-3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask�
Emodel_6/transformer_encoder_20/Encoder-2nd-NormalizationLayer-3/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
Dmodel_6/transformer_encoder_20/Encoder-2nd-NormalizationLayer-3/ProdProdVmodel_6/transformer_encoder_20/Encoder-2nd-NormalizationLayer-3/strided_slice:output:0Nmodel_6/transformer_encoder_20/Encoder-2nd-NormalizationLayer-3/Const:output:0*
T0*
_output_shapes
: �
Umodel_6/transformer_encoder_20/Encoder-2nd-NormalizationLayer-3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:�
Wmodel_6/transformer_encoder_20/Encoder-2nd-NormalizationLayer-3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: �
Wmodel_6/transformer_encoder_20/Encoder-2nd-NormalizationLayer-3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
Omodel_6/transformer_encoder_20/Encoder-2nd-NormalizationLayer-3/strided_slice_1StridedSliceNmodel_6/transformer_encoder_20/Encoder-2nd-NormalizationLayer-3/Shape:output:0^model_6/transformer_encoder_20/Encoder-2nd-NormalizationLayer-3/strided_slice_1/stack:output:0`model_6/transformer_encoder_20/Encoder-2nd-NormalizationLayer-3/strided_slice_1/stack_1:output:0`model_6/transformer_encoder_20/Encoder-2nd-NormalizationLayer-3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask�
Gmodel_6/transformer_encoder_20/Encoder-2nd-NormalizationLayer-3/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
Fmodel_6/transformer_encoder_20/Encoder-2nd-NormalizationLayer-3/Prod_1ProdXmodel_6/transformer_encoder_20/Encoder-2nd-NormalizationLayer-3/strided_slice_1:output:0Pmodel_6/transformer_encoder_20/Encoder-2nd-NormalizationLayer-3/Const_1:output:0*
T0*
_output_shapes
: �
Omodel_6/transformer_encoder_20/Encoder-2nd-NormalizationLayer-3/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :�
Omodel_6/transformer_encoder_20/Encoder-2nd-NormalizationLayer-3/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :�
Mmodel_6/transformer_encoder_20/Encoder-2nd-NormalizationLayer-3/Reshape/shapePackXmodel_6/transformer_encoder_20/Encoder-2nd-NormalizationLayer-3/Reshape/shape/0:output:0Mmodel_6/transformer_encoder_20/Encoder-2nd-NormalizationLayer-3/Prod:output:0Omodel_6/transformer_encoder_20/Encoder-2nd-NormalizationLayer-3/Prod_1:output:0Xmodel_6/transformer_encoder_20/Encoder-2nd-NormalizationLayer-3/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:�
Gmodel_6/transformer_encoder_20/Encoder-2nd-NormalizationLayer-3/ReshapeReshapeBmodel_6/transformer_encoder_20/Encoder-1st-AdditionLayer-3/add:z:0Vmodel_6/transformer_encoder_20/Encoder-2nd-NormalizationLayer-3/Reshape/shape:output:0*
T0*8
_output_shapes&
$:"�������������������
Kmodel_6/transformer_encoder_20/Encoder-2nd-NormalizationLayer-3/ones/packedPackMmodel_6/transformer_encoder_20/Encoder-2nd-NormalizationLayer-3/Prod:output:0*
N*
T0*
_output_shapes
:�
Jmodel_6/transformer_encoder_20/Encoder-2nd-NormalizationLayer-3/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
Dmodel_6/transformer_encoder_20/Encoder-2nd-NormalizationLayer-3/onesFillTmodel_6/transformer_encoder_20/Encoder-2nd-NormalizationLayer-3/ones/packed:output:0Smodel_6/transformer_encoder_20/Encoder-2nd-NormalizationLayer-3/ones/Const:output:0*
T0*#
_output_shapes
:����������
Lmodel_6/transformer_encoder_20/Encoder-2nd-NormalizationLayer-3/zeros/packedPackMmodel_6/transformer_encoder_20/Encoder-2nd-NormalizationLayer-3/Prod:output:0*
N*
T0*
_output_shapes
:�
Kmodel_6/transformer_encoder_20/Encoder-2nd-NormalizationLayer-3/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
Emodel_6/transformer_encoder_20/Encoder-2nd-NormalizationLayer-3/zerosFillUmodel_6/transformer_encoder_20/Encoder-2nd-NormalizationLayer-3/zeros/packed:output:0Tmodel_6/transformer_encoder_20/Encoder-2nd-NormalizationLayer-3/zeros/Const:output:0*
T0*#
_output_shapes
:����������
Gmodel_6/transformer_encoder_20/Encoder-2nd-NormalizationLayer-3/Const_2Const*
_output_shapes
: *
dtype0*
valueB �
Gmodel_6/transformer_encoder_20/Encoder-2nd-NormalizationLayer-3/Const_3Const*
_output_shapes
: *
dtype0*
valueB �
Pmodel_6/transformer_encoder_20/Encoder-2nd-NormalizationLayer-3/FusedBatchNormV3FusedBatchNormV3Pmodel_6/transformer_encoder_20/Encoder-2nd-NormalizationLayer-3/Reshape:output:0Mmodel_6/transformer_encoder_20/Encoder-2nd-NormalizationLayer-3/ones:output:0Nmodel_6/transformer_encoder_20/Encoder-2nd-NormalizationLayer-3/zeros:output:0Pmodel_6/transformer_encoder_20/Encoder-2nd-NormalizationLayer-3/Const_2:output:0Pmodel_6/transformer_encoder_20/Encoder-2nd-NormalizationLayer-3/Const_3:output:0*
T0*
U0*x
_output_shapesf
d:"������������������:���������:���������:���������:���������:*
data_formatNCHW*
epsilon%o�:�
Imodel_6/transformer_encoder_20/Encoder-2nd-NormalizationLayer-3/Reshape_1ReshapeTmodel_6/transformer_encoder_20/Encoder-2nd-NormalizationLayer-3/FusedBatchNormV3:y:0Nmodel_6/transformer_encoder_20/Encoder-2nd-NormalizationLayer-3/Shape:output:0*
T0*+
_output_shapes
:���������P	�
Rmodel_6/transformer_encoder_20/Encoder-2nd-NormalizationLayer-3/mul/ReadVariableOpReadVariableOp[model_6_transformer_encoder_20_encoder_2nd_normalizationlayer_3_mul_readvariableop_resource*
_output_shapes
:	*
dtype0�
Cmodel_6/transformer_encoder_20/Encoder-2nd-NormalizationLayer-3/mulMulRmodel_6/transformer_encoder_20/Encoder-2nd-NormalizationLayer-3/Reshape_1:output:0Zmodel_6/transformer_encoder_20/Encoder-2nd-NormalizationLayer-3/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������P	�
Rmodel_6/transformer_encoder_20/Encoder-2nd-NormalizationLayer-3/add/ReadVariableOpReadVariableOp[model_6_transformer_encoder_20_encoder_2nd_normalizationlayer_3_add_readvariableop_resource*
_output_shapes
:	*
dtype0�
Cmodel_6/transformer_encoder_20/Encoder-2nd-NormalizationLayer-3/addAddV2Gmodel_6/transformer_encoder_20/Encoder-2nd-NormalizationLayer-3/mul:z:0Zmodel_6/transformer_encoder_20/Encoder-2nd-NormalizationLayer-3/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������P	�
Tmodel_6/transformer_encoder_20/Encoder-FeedForwardLayer_1_3/Tensordot/ReadVariableOpReadVariableOp]model_6_transformer_encoder_20_encoder_feedforwardlayer_1_3_tensordot_readvariableop_resource*
_output_shapes

:		*
dtype0�
Jmodel_6/transformer_encoder_20/Encoder-FeedForwardLayer_1_3/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:�
Jmodel_6/transformer_encoder_20/Encoder-FeedForwardLayer_1_3/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       �
Kmodel_6/transformer_encoder_20/Encoder-FeedForwardLayer_1_3/Tensordot/ShapeShapeGmodel_6/transformer_encoder_20/Encoder-2nd-NormalizationLayer-3/add:z:0*
T0*
_output_shapes
::���
Smodel_6/transformer_encoder_20/Encoder-FeedForwardLayer_1_3/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Nmodel_6/transformer_encoder_20/Encoder-FeedForwardLayer_1_3/Tensordot/GatherV2GatherV2Tmodel_6/transformer_encoder_20/Encoder-FeedForwardLayer_1_3/Tensordot/Shape:output:0Smodel_6/transformer_encoder_20/Encoder-FeedForwardLayer_1_3/Tensordot/free:output:0\model_6/transformer_encoder_20/Encoder-FeedForwardLayer_1_3/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
Umodel_6/transformer_encoder_20/Encoder-FeedForwardLayer_1_3/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Pmodel_6/transformer_encoder_20/Encoder-FeedForwardLayer_1_3/Tensordot/GatherV2_1GatherV2Tmodel_6/transformer_encoder_20/Encoder-FeedForwardLayer_1_3/Tensordot/Shape:output:0Smodel_6/transformer_encoder_20/Encoder-FeedForwardLayer_1_3/Tensordot/axes:output:0^model_6/transformer_encoder_20/Encoder-FeedForwardLayer_1_3/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
Kmodel_6/transformer_encoder_20/Encoder-FeedForwardLayer_1_3/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
Jmodel_6/transformer_encoder_20/Encoder-FeedForwardLayer_1_3/Tensordot/ProdProdWmodel_6/transformer_encoder_20/Encoder-FeedForwardLayer_1_3/Tensordot/GatherV2:output:0Tmodel_6/transformer_encoder_20/Encoder-FeedForwardLayer_1_3/Tensordot/Const:output:0*
T0*
_output_shapes
: �
Mmodel_6/transformer_encoder_20/Encoder-FeedForwardLayer_1_3/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
Lmodel_6/transformer_encoder_20/Encoder-FeedForwardLayer_1_3/Tensordot/Prod_1ProdYmodel_6/transformer_encoder_20/Encoder-FeedForwardLayer_1_3/Tensordot/GatherV2_1:output:0Vmodel_6/transformer_encoder_20/Encoder-FeedForwardLayer_1_3/Tensordot/Const_1:output:0*
T0*
_output_shapes
: �
Qmodel_6/transformer_encoder_20/Encoder-FeedForwardLayer_1_3/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Lmodel_6/transformer_encoder_20/Encoder-FeedForwardLayer_1_3/Tensordot/concatConcatV2Smodel_6/transformer_encoder_20/Encoder-FeedForwardLayer_1_3/Tensordot/free:output:0Smodel_6/transformer_encoder_20/Encoder-FeedForwardLayer_1_3/Tensordot/axes:output:0Zmodel_6/transformer_encoder_20/Encoder-FeedForwardLayer_1_3/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
Kmodel_6/transformer_encoder_20/Encoder-FeedForwardLayer_1_3/Tensordot/stackPackSmodel_6/transformer_encoder_20/Encoder-FeedForwardLayer_1_3/Tensordot/Prod:output:0Umodel_6/transformer_encoder_20/Encoder-FeedForwardLayer_1_3/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
Omodel_6/transformer_encoder_20/Encoder-FeedForwardLayer_1_3/Tensordot/transpose	TransposeGmodel_6/transformer_encoder_20/Encoder-2nd-NormalizationLayer-3/add:z:0Umodel_6/transformer_encoder_20/Encoder-FeedForwardLayer_1_3/Tensordot/concat:output:0*
T0*+
_output_shapes
:���������P	�
Mmodel_6/transformer_encoder_20/Encoder-FeedForwardLayer_1_3/Tensordot/ReshapeReshapeSmodel_6/transformer_encoder_20/Encoder-FeedForwardLayer_1_3/Tensordot/transpose:y:0Tmodel_6/transformer_encoder_20/Encoder-FeedForwardLayer_1_3/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
Lmodel_6/transformer_encoder_20/Encoder-FeedForwardLayer_1_3/Tensordot/MatMulMatMulVmodel_6/transformer_encoder_20/Encoder-FeedForwardLayer_1_3/Tensordot/Reshape:output:0\model_6/transformer_encoder_20/Encoder-FeedForwardLayer_1_3/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������	�
Mmodel_6/transformer_encoder_20/Encoder-FeedForwardLayer_1_3/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:	�
Smodel_6/transformer_encoder_20/Encoder-FeedForwardLayer_1_3/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Nmodel_6/transformer_encoder_20/Encoder-FeedForwardLayer_1_3/Tensordot/concat_1ConcatV2Wmodel_6/transformer_encoder_20/Encoder-FeedForwardLayer_1_3/Tensordot/GatherV2:output:0Vmodel_6/transformer_encoder_20/Encoder-FeedForwardLayer_1_3/Tensordot/Const_2:output:0\model_6/transformer_encoder_20/Encoder-FeedForwardLayer_1_3/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
Emodel_6/transformer_encoder_20/Encoder-FeedForwardLayer_1_3/TensordotReshapeVmodel_6/transformer_encoder_20/Encoder-FeedForwardLayer_1_3/Tensordot/MatMul:product:0Wmodel_6/transformer_encoder_20/Encoder-FeedForwardLayer_1_3/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������P	�
Rmodel_6/transformer_encoder_20/Encoder-FeedForwardLayer_1_3/BiasAdd/ReadVariableOpReadVariableOp[model_6_transformer_encoder_20_encoder_feedforwardlayer_1_3_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype0�
Cmodel_6/transformer_encoder_20/Encoder-FeedForwardLayer_1_3/BiasAddBiasAddNmodel_6/transformer_encoder_20/Encoder-FeedForwardLayer_1_3/Tensordot:output:0Zmodel_6/transformer_encoder_20/Encoder-FeedForwardLayer_1_3/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������P	�
Fmodel_6/transformer_encoder_20/Encoder-FeedForwardLayer_1_3/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
Dmodel_6/transformer_encoder_20/Encoder-FeedForwardLayer_1_3/Gelu/mulMulOmodel_6/transformer_encoder_20/Encoder-FeedForwardLayer_1_3/Gelu/mul/x:output:0Lmodel_6/transformer_encoder_20/Encoder-FeedForwardLayer_1_3/BiasAdd:output:0*
T0*+
_output_shapes
:���������P	�
Gmodel_6/transformer_encoder_20/Encoder-FeedForwardLayer_1_3/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *��?�
Hmodel_6/transformer_encoder_20/Encoder-FeedForwardLayer_1_3/Gelu/truedivRealDivLmodel_6/transformer_encoder_20/Encoder-FeedForwardLayer_1_3/BiasAdd:output:0Pmodel_6/transformer_encoder_20/Encoder-FeedForwardLayer_1_3/Gelu/Cast/x:output:0*
T0*+
_output_shapes
:���������P	�
Dmodel_6/transformer_encoder_20/Encoder-FeedForwardLayer_1_3/Gelu/ErfErfLmodel_6/transformer_encoder_20/Encoder-FeedForwardLayer_1_3/Gelu/truediv:z:0*
T0*+
_output_shapes
:���������P	�
Fmodel_6/transformer_encoder_20/Encoder-FeedForwardLayer_1_3/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
Dmodel_6/transformer_encoder_20/Encoder-FeedForwardLayer_1_3/Gelu/addAddV2Omodel_6/transformer_encoder_20/Encoder-FeedForwardLayer_1_3/Gelu/add/x:output:0Hmodel_6/transformer_encoder_20/Encoder-FeedForwardLayer_1_3/Gelu/Erf:y:0*
T0*+
_output_shapes
:���������P	�
Fmodel_6/transformer_encoder_20/Encoder-FeedForwardLayer_1_3/Gelu/mul_1MulHmodel_6/transformer_encoder_20/Encoder-FeedForwardLayer_1_3/Gelu/mul:z:0Hmodel_6/transformer_encoder_20/Encoder-FeedForwardLayer_1_3/Gelu/add:z:0*
T0*+
_output_shapes
:���������P	�
Tmodel_6/transformer_encoder_20/Encoder-FeedForwardLayer_2_3/Tensordot/ReadVariableOpReadVariableOp]model_6_transformer_encoder_20_encoder_feedforwardlayer_2_3_tensordot_readvariableop_resource*
_output_shapes

:		*
dtype0�
Jmodel_6/transformer_encoder_20/Encoder-FeedForwardLayer_2_3/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:�
Jmodel_6/transformer_encoder_20/Encoder-FeedForwardLayer_2_3/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       �
Kmodel_6/transformer_encoder_20/Encoder-FeedForwardLayer_2_3/Tensordot/ShapeShapeJmodel_6/transformer_encoder_20/Encoder-FeedForwardLayer_1_3/Gelu/mul_1:z:0*
T0*
_output_shapes
::���
Smodel_6/transformer_encoder_20/Encoder-FeedForwardLayer_2_3/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Nmodel_6/transformer_encoder_20/Encoder-FeedForwardLayer_2_3/Tensordot/GatherV2GatherV2Tmodel_6/transformer_encoder_20/Encoder-FeedForwardLayer_2_3/Tensordot/Shape:output:0Smodel_6/transformer_encoder_20/Encoder-FeedForwardLayer_2_3/Tensordot/free:output:0\model_6/transformer_encoder_20/Encoder-FeedForwardLayer_2_3/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
Umodel_6/transformer_encoder_20/Encoder-FeedForwardLayer_2_3/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Pmodel_6/transformer_encoder_20/Encoder-FeedForwardLayer_2_3/Tensordot/GatherV2_1GatherV2Tmodel_6/transformer_encoder_20/Encoder-FeedForwardLayer_2_3/Tensordot/Shape:output:0Smodel_6/transformer_encoder_20/Encoder-FeedForwardLayer_2_3/Tensordot/axes:output:0^model_6/transformer_encoder_20/Encoder-FeedForwardLayer_2_3/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
Kmodel_6/transformer_encoder_20/Encoder-FeedForwardLayer_2_3/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
Jmodel_6/transformer_encoder_20/Encoder-FeedForwardLayer_2_3/Tensordot/ProdProdWmodel_6/transformer_encoder_20/Encoder-FeedForwardLayer_2_3/Tensordot/GatherV2:output:0Tmodel_6/transformer_encoder_20/Encoder-FeedForwardLayer_2_3/Tensordot/Const:output:0*
T0*
_output_shapes
: �
Mmodel_6/transformer_encoder_20/Encoder-FeedForwardLayer_2_3/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
Lmodel_6/transformer_encoder_20/Encoder-FeedForwardLayer_2_3/Tensordot/Prod_1ProdYmodel_6/transformer_encoder_20/Encoder-FeedForwardLayer_2_3/Tensordot/GatherV2_1:output:0Vmodel_6/transformer_encoder_20/Encoder-FeedForwardLayer_2_3/Tensordot/Const_1:output:0*
T0*
_output_shapes
: �
Qmodel_6/transformer_encoder_20/Encoder-FeedForwardLayer_2_3/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Lmodel_6/transformer_encoder_20/Encoder-FeedForwardLayer_2_3/Tensordot/concatConcatV2Smodel_6/transformer_encoder_20/Encoder-FeedForwardLayer_2_3/Tensordot/free:output:0Smodel_6/transformer_encoder_20/Encoder-FeedForwardLayer_2_3/Tensordot/axes:output:0Zmodel_6/transformer_encoder_20/Encoder-FeedForwardLayer_2_3/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
Kmodel_6/transformer_encoder_20/Encoder-FeedForwardLayer_2_3/Tensordot/stackPackSmodel_6/transformer_encoder_20/Encoder-FeedForwardLayer_2_3/Tensordot/Prod:output:0Umodel_6/transformer_encoder_20/Encoder-FeedForwardLayer_2_3/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
Omodel_6/transformer_encoder_20/Encoder-FeedForwardLayer_2_3/Tensordot/transpose	TransposeJmodel_6/transformer_encoder_20/Encoder-FeedForwardLayer_1_3/Gelu/mul_1:z:0Umodel_6/transformer_encoder_20/Encoder-FeedForwardLayer_2_3/Tensordot/concat:output:0*
T0*+
_output_shapes
:���������P	�
Mmodel_6/transformer_encoder_20/Encoder-FeedForwardLayer_2_3/Tensordot/ReshapeReshapeSmodel_6/transformer_encoder_20/Encoder-FeedForwardLayer_2_3/Tensordot/transpose:y:0Tmodel_6/transformer_encoder_20/Encoder-FeedForwardLayer_2_3/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
Lmodel_6/transformer_encoder_20/Encoder-FeedForwardLayer_2_3/Tensordot/MatMulMatMulVmodel_6/transformer_encoder_20/Encoder-FeedForwardLayer_2_3/Tensordot/Reshape:output:0\model_6/transformer_encoder_20/Encoder-FeedForwardLayer_2_3/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������	�
Mmodel_6/transformer_encoder_20/Encoder-FeedForwardLayer_2_3/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:	�
Smodel_6/transformer_encoder_20/Encoder-FeedForwardLayer_2_3/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Nmodel_6/transformer_encoder_20/Encoder-FeedForwardLayer_2_3/Tensordot/concat_1ConcatV2Wmodel_6/transformer_encoder_20/Encoder-FeedForwardLayer_2_3/Tensordot/GatherV2:output:0Vmodel_6/transformer_encoder_20/Encoder-FeedForwardLayer_2_3/Tensordot/Const_2:output:0\model_6/transformer_encoder_20/Encoder-FeedForwardLayer_2_3/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
Emodel_6/transformer_encoder_20/Encoder-FeedForwardLayer_2_3/TensordotReshapeVmodel_6/transformer_encoder_20/Encoder-FeedForwardLayer_2_3/Tensordot/MatMul:product:0Wmodel_6/transformer_encoder_20/Encoder-FeedForwardLayer_2_3/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������P	�
Rmodel_6/transformer_encoder_20/Encoder-FeedForwardLayer_2_3/BiasAdd/ReadVariableOpReadVariableOp[model_6_transformer_encoder_20_encoder_feedforwardlayer_2_3_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype0�
Cmodel_6/transformer_encoder_20/Encoder-FeedForwardLayer_2_3/BiasAddBiasAddNmodel_6/transformer_encoder_20/Encoder-FeedForwardLayer_2_3/Tensordot:output:0Zmodel_6/transformer_encoder_20/Encoder-FeedForwardLayer_2_3/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������P	�
2model_6/transformer_encoder_20/dropout_20/IdentityIdentityLmodel_6/transformer_encoder_20/Encoder-FeedForwardLayer_2_3/BiasAdd:output:0*
T0*+
_output_shapes
:���������P	�
>model_6/transformer_encoder_20/Encoder-2nd-AdditionLayer-3/addAddV2Bmodel_6/transformer_encoder_20/Encoder-1st-AdditionLayer-3/add:z:0;model_6/transformer_encoder_20/dropout_20/Identity:output:0*
T0*+
_output_shapes
:���������P	�
model_6/FinalLayerNorm/ShapeShapeBmodel_6/transformer_encoder_20/Encoder-2nd-AdditionLayer-3/add:z:0*
T0*
_output_shapes
::��t
*model_6/FinalLayerNorm/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: v
,model_6/FinalLayerNorm/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:v
,model_6/FinalLayerNorm/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
$model_6/FinalLayerNorm/strided_sliceStridedSlice%model_6/FinalLayerNorm/Shape:output:03model_6/FinalLayerNorm/strided_slice/stack:output:05model_6/FinalLayerNorm/strided_slice/stack_1:output:05model_6/FinalLayerNorm/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskf
model_6/FinalLayerNorm/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
model_6/FinalLayerNorm/ProdProd-model_6/FinalLayerNorm/strided_slice:output:0%model_6/FinalLayerNorm/Const:output:0*
T0*
_output_shapes
: v
,model_6/FinalLayerNorm/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:x
.model_6/FinalLayerNorm/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: x
.model_6/FinalLayerNorm/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
&model_6/FinalLayerNorm/strided_slice_1StridedSlice%model_6/FinalLayerNorm/Shape:output:05model_6/FinalLayerNorm/strided_slice_1/stack:output:07model_6/FinalLayerNorm/strided_slice_1/stack_1:output:07model_6/FinalLayerNorm/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_maskh
model_6/FinalLayerNorm/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
model_6/FinalLayerNorm/Prod_1Prod/model_6/FinalLayerNorm/strided_slice_1:output:0'model_6/FinalLayerNorm/Const_1:output:0*
T0*
_output_shapes
: h
&model_6/FinalLayerNorm/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :h
&model_6/FinalLayerNorm/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :�
$model_6/FinalLayerNorm/Reshape/shapePack/model_6/FinalLayerNorm/Reshape/shape/0:output:0$model_6/FinalLayerNorm/Prod:output:0&model_6/FinalLayerNorm/Prod_1:output:0/model_6/FinalLayerNorm/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:�
model_6/FinalLayerNorm/ReshapeReshapeBmodel_6/transformer_encoder_20/Encoder-2nd-AdditionLayer-3/add:z:0-model_6/FinalLayerNorm/Reshape/shape:output:0*
T0*8
_output_shapes&
$:"������������������~
"model_6/FinalLayerNorm/ones/packedPack$model_6/FinalLayerNorm/Prod:output:0*
N*
T0*
_output_shapes
:f
!model_6/FinalLayerNorm/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
model_6/FinalLayerNorm/onesFill+model_6/FinalLayerNorm/ones/packed:output:0*model_6/FinalLayerNorm/ones/Const:output:0*
T0*#
_output_shapes
:���������
#model_6/FinalLayerNorm/zeros/packedPack$model_6/FinalLayerNorm/Prod:output:0*
N*
T0*
_output_shapes
:g
"model_6/FinalLayerNorm/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
model_6/FinalLayerNorm/zerosFill,model_6/FinalLayerNorm/zeros/packed:output:0+model_6/FinalLayerNorm/zeros/Const:output:0*
T0*#
_output_shapes
:���������a
model_6/FinalLayerNorm/Const_2Const*
_output_shapes
: *
dtype0*
valueB a
model_6/FinalLayerNorm/Const_3Const*
_output_shapes
: *
dtype0*
valueB �
'model_6/FinalLayerNorm/FusedBatchNormV3FusedBatchNormV3'model_6/FinalLayerNorm/Reshape:output:0$model_6/FinalLayerNorm/ones:output:0%model_6/FinalLayerNorm/zeros:output:0'model_6/FinalLayerNorm/Const_2:output:0'model_6/FinalLayerNorm/Const_3:output:0*
T0*
U0*x
_output_shapesf
d:"������������������:���������:���������:���������:���������:*
data_formatNCHW*
epsilon%o�:�
 model_6/FinalLayerNorm/Reshape_1Reshape+model_6/FinalLayerNorm/FusedBatchNormV3:y:0%model_6/FinalLayerNorm/Shape:output:0*
T0*+
_output_shapes
:���������P	�
)model_6/FinalLayerNorm/mul/ReadVariableOpReadVariableOp2model_6_finallayernorm_mul_readvariableop_resource*
_output_shapes
:	*
dtype0�
model_6/FinalLayerNorm/mulMul)model_6/FinalLayerNorm/Reshape_1:output:01model_6/FinalLayerNorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������P	�
)model_6/FinalLayerNorm/add/ReadVariableOpReadVariableOp2model_6_finallayernorm_add_readvariableop_resource*
_output_shapes
:	*
dtype0�
model_6/FinalLayerNorm/addAddV2model_6/FinalLayerNorm/mul:z:01model_6/FinalLayerNorm/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������P	�
>model_6/ReduceStackDimensionViaSummation/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :�
,model_6/ReduceStackDimensionViaSummation/SumSummodel_6/FinalLayerNorm/add:z:0Gmodel_6/ReduceStackDimensionViaSummation/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:���������	w
2model_6/ReduceStackDimensionViaSummation/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �B�
0model_6/ReduceStackDimensionViaSummation/truedivRealDiv5model_6/ReduceStackDimensionViaSummation/Sum:output:0;model_6/ReduceStackDimensionViaSummation/truediv/y:output:0*
T0*'
_output_shapes
:���������	z
!model_6/StandardizeTimeLimit/CastCasttimelimitinput*

DstT0*

SrcT0*'
_output_shapes
:���������g
"model_6/StandardizeTimeLimit/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  pA�
 model_6/StandardizeTimeLimit/subSub%model_6/StandardizeTimeLimit/Cast:y:0+model_6/StandardizeTimeLimit/sub/y:output:0*
T0*'
_output_shapes
:���������k
&model_6/StandardizeTimeLimit/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �@�
$model_6/StandardizeTimeLimit/truedivRealDiv$model_6/StandardizeTimeLimit/sub:z:0/model_6/StandardizeTimeLimit/truediv/y:output:0*
T0*'
_output_shapes
:���������f
$model_6/ConcatenateLayer/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
model_6/ConcatenateLayer/concatConcatV24model_6/ReduceStackDimensionViaSummation/truediv:z:0(model_6/StandardizeTimeLimit/truediv:z:0-model_6/ConcatenateLayer/concat/axis:output:0*
N*
T0*'
_output_shapes
:���������
�
:model_6/FullyConnectedLayerAreaRatio/MatMul/ReadVariableOpReadVariableOpCmodel_6_fullyconnectedlayerarearatio_matmul_readvariableop_resource*
_output_shapes

:

*
dtype0�
+model_6/FullyConnectedLayerAreaRatio/MatMulMatMul(model_6/ConcatenateLayer/concat:output:0Bmodel_6/FullyConnectedLayerAreaRatio/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
�
;model_6/FullyConnectedLayerAreaRatio/BiasAdd/ReadVariableOpReadVariableOpDmodel_6_fullyconnectedlayerarearatio_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0�
,model_6/FullyConnectedLayerAreaRatio/BiasAddBiasAdd5model_6/FullyConnectedLayerAreaRatio/MatMul:product:0Cmodel_6/FullyConnectedLayerAreaRatio/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
t
/model_6/FullyConnectedLayerAreaRatio/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
-model_6/FullyConnectedLayerAreaRatio/Gelu/mulMul8model_6/FullyConnectedLayerAreaRatio/Gelu/mul/x:output:05model_6/FullyConnectedLayerAreaRatio/BiasAdd:output:0*
T0*'
_output_shapes
:���������
u
0model_6/FullyConnectedLayerAreaRatio/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *��?�
1model_6/FullyConnectedLayerAreaRatio/Gelu/truedivRealDiv5model_6/FullyConnectedLayerAreaRatio/BiasAdd:output:09model_6/FullyConnectedLayerAreaRatio/Gelu/Cast/x:output:0*
T0*'
_output_shapes
:���������
�
-model_6/FullyConnectedLayerAreaRatio/Gelu/ErfErf5model_6/FullyConnectedLayerAreaRatio/Gelu/truediv:z:0*
T0*'
_output_shapes
:���������
t
/model_6/FullyConnectedLayerAreaRatio/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
-model_6/FullyConnectedLayerAreaRatio/Gelu/addAddV28model_6/FullyConnectedLayerAreaRatio/Gelu/add/x:output:01model_6/FullyConnectedLayerAreaRatio/Gelu/Erf:y:0*
T0*'
_output_shapes
:���������
�
/model_6/FullyConnectedLayerAreaRatio/Gelu/mul_1Mul1model_6/FullyConnectedLayerAreaRatio/Gelu/mul:z:01model_6/FullyConnectedLayerAreaRatio/Gelu/add:z:0*
T0*'
_output_shapes
:���������
�
1model_6/PredictionAreaRatio/MatMul/ReadVariableOpReadVariableOp:model_6_predictionarearatio_matmul_readvariableop_resource*
_output_shapes

:
*
dtype0�
"model_6/PredictionAreaRatio/MatMulMatMul3model_6/FullyConnectedLayerAreaRatio/Gelu/mul_1:z:09model_6/PredictionAreaRatio/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
2model_6/PredictionAreaRatio/BiasAdd/ReadVariableOpReadVariableOp;model_6_predictionarearatio_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
#model_6/PredictionAreaRatio/BiasAddBiasAdd,model_6/PredictionAreaRatio/MatMul:product:0:model_6/PredictionAreaRatio/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
#model_6/PredictionAreaRatio/SigmoidSigmoid,model_6/PredictionAreaRatio/BiasAdd:output:0*
T0*'
_output_shapes
:���������m
model_6/Output/SqueezeSqueeze'model_6/PredictionAreaRatio/Sigmoid:y:0*
T0*
_output_shapes
:_
IdentityIdentitymodel_6/Output/Squeeze:output:0^NoOp*
T0*
_output_shapes
:�$
NoOpNoOp*^model_6/FinalLayerNorm/add/ReadVariableOp*^model_6/FinalLayerNorm/mul/ReadVariableOp<^model_6/FullyConnectedLayerAreaRatio/BiasAdd/ReadVariableOp;^model_6/FullyConnectedLayerAreaRatio/MatMul/ReadVariableOp3^model_6/PredictionAreaRatio/BiasAdd/ReadVariableOp2^model_6/PredictionAreaRatio/MatMul/ReadVariableOpS^model_6/transformer_encoder_18/Encoder-1st-NormalizationLayer-1/add/ReadVariableOpS^model_6/transformer_encoder_18/Encoder-1st-NormalizationLayer-1/mul/ReadVariableOpS^model_6/transformer_encoder_18/Encoder-2nd-NormalizationLayer-1/add/ReadVariableOpS^model_6/transformer_encoder_18/Encoder-2nd-NormalizationLayer-1/mul/ReadVariableOpS^model_6/transformer_encoder_18/Encoder-FeedForwardLayer_1_1/BiasAdd/ReadVariableOpU^model_6/transformer_encoder_18/Encoder-FeedForwardLayer_1_1/Tensordot/ReadVariableOpS^model_6/transformer_encoder_18/Encoder-FeedForwardLayer_2_1/BiasAdd/ReadVariableOpU^model_6/transformer_encoder_18/Encoder-FeedForwardLayer_2_1/Tensordot/ReadVariableOp`^model_6/transformer_encoder_18/Encoder-SelfAttentionLayer-1/attention_output/add/ReadVariableOpj^model_6/transformer_encoder_18/Encoder-SelfAttentionLayer-1/attention_output/einsum/Einsum/ReadVariableOpS^model_6/transformer_encoder_18/Encoder-SelfAttentionLayer-1/key/add/ReadVariableOp]^model_6/transformer_encoder_18/Encoder-SelfAttentionLayer-1/key/einsum/Einsum/ReadVariableOpU^model_6/transformer_encoder_18/Encoder-SelfAttentionLayer-1/query/add/ReadVariableOp_^model_6/transformer_encoder_18/Encoder-SelfAttentionLayer-1/query/einsum/Einsum/ReadVariableOpU^model_6/transformer_encoder_18/Encoder-SelfAttentionLayer-1/value/add/ReadVariableOp_^model_6/transformer_encoder_18/Encoder-SelfAttentionLayer-1/value/einsum/Einsum/ReadVariableOpS^model_6/transformer_encoder_19/Encoder-1st-NormalizationLayer-2/add/ReadVariableOpS^model_6/transformer_encoder_19/Encoder-1st-NormalizationLayer-2/mul/ReadVariableOpS^model_6/transformer_encoder_19/Encoder-2nd-NormalizationLayer-2/add/ReadVariableOpS^model_6/transformer_encoder_19/Encoder-2nd-NormalizationLayer-2/mul/ReadVariableOpS^model_6/transformer_encoder_19/Encoder-FeedForwardLayer_1_2/BiasAdd/ReadVariableOpU^model_6/transformer_encoder_19/Encoder-FeedForwardLayer_1_2/Tensordot/ReadVariableOpS^model_6/transformer_encoder_19/Encoder-FeedForwardLayer_2_2/BiasAdd/ReadVariableOpU^model_6/transformer_encoder_19/Encoder-FeedForwardLayer_2_2/Tensordot/ReadVariableOp`^model_6/transformer_encoder_19/Encoder-SelfAttentionLayer-2/attention_output/add/ReadVariableOpj^model_6/transformer_encoder_19/Encoder-SelfAttentionLayer-2/attention_output/einsum/Einsum/ReadVariableOpS^model_6/transformer_encoder_19/Encoder-SelfAttentionLayer-2/key/add/ReadVariableOp]^model_6/transformer_encoder_19/Encoder-SelfAttentionLayer-2/key/einsum/Einsum/ReadVariableOpU^model_6/transformer_encoder_19/Encoder-SelfAttentionLayer-2/query/add/ReadVariableOp_^model_6/transformer_encoder_19/Encoder-SelfAttentionLayer-2/query/einsum/Einsum/ReadVariableOpU^model_6/transformer_encoder_19/Encoder-SelfAttentionLayer-2/value/add/ReadVariableOp_^model_6/transformer_encoder_19/Encoder-SelfAttentionLayer-2/value/einsum/Einsum/ReadVariableOpS^model_6/transformer_encoder_20/Encoder-1st-NormalizationLayer-3/add/ReadVariableOpS^model_6/transformer_encoder_20/Encoder-1st-NormalizationLayer-3/mul/ReadVariableOpS^model_6/transformer_encoder_20/Encoder-2nd-NormalizationLayer-3/add/ReadVariableOpS^model_6/transformer_encoder_20/Encoder-2nd-NormalizationLayer-3/mul/ReadVariableOpS^model_6/transformer_encoder_20/Encoder-FeedForwardLayer_1_3/BiasAdd/ReadVariableOpU^model_6/transformer_encoder_20/Encoder-FeedForwardLayer_1_3/Tensordot/ReadVariableOpS^model_6/transformer_encoder_20/Encoder-FeedForwardLayer_2_3/BiasAdd/ReadVariableOpU^model_6/transformer_encoder_20/Encoder-FeedForwardLayer_2_3/Tensordot/ReadVariableOp`^model_6/transformer_encoder_20/Encoder-SelfAttentionLayer-3/attention_output/add/ReadVariableOpj^model_6/transformer_encoder_20/Encoder-SelfAttentionLayer-3/attention_output/einsum/Einsum/ReadVariableOpS^model_6/transformer_encoder_20/Encoder-SelfAttentionLayer-3/key/add/ReadVariableOp]^model_6/transformer_encoder_20/Encoder-SelfAttentionLayer-3/key/einsum/Einsum/ReadVariableOpU^model_6/transformer_encoder_20/Encoder-SelfAttentionLayer-3/query/add/ReadVariableOp_^model_6/transformer_encoder_20/Encoder-SelfAttentionLayer-3/query/einsum/Einsum/ReadVariableOpU^model_6/transformer_encoder_20/Encoder-SelfAttentionLayer-3/value/add/ReadVariableOp_^model_6/transformer_encoder_20/Encoder-SelfAttentionLayer-3/value/einsum/Einsum/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:���������P	:���������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2V
)model_6/FinalLayerNorm/add/ReadVariableOp)model_6/FinalLayerNorm/add/ReadVariableOp2V
)model_6/FinalLayerNorm/mul/ReadVariableOp)model_6/FinalLayerNorm/mul/ReadVariableOp2z
;model_6/FullyConnectedLayerAreaRatio/BiasAdd/ReadVariableOp;model_6/FullyConnectedLayerAreaRatio/BiasAdd/ReadVariableOp2x
:model_6/FullyConnectedLayerAreaRatio/MatMul/ReadVariableOp:model_6/FullyConnectedLayerAreaRatio/MatMul/ReadVariableOp2h
2model_6/PredictionAreaRatio/BiasAdd/ReadVariableOp2model_6/PredictionAreaRatio/BiasAdd/ReadVariableOp2f
1model_6/PredictionAreaRatio/MatMul/ReadVariableOp1model_6/PredictionAreaRatio/MatMul/ReadVariableOp2�
Rmodel_6/transformer_encoder_18/Encoder-1st-NormalizationLayer-1/add/ReadVariableOpRmodel_6/transformer_encoder_18/Encoder-1st-NormalizationLayer-1/add/ReadVariableOp2�
Rmodel_6/transformer_encoder_18/Encoder-1st-NormalizationLayer-1/mul/ReadVariableOpRmodel_6/transformer_encoder_18/Encoder-1st-NormalizationLayer-1/mul/ReadVariableOp2�
Rmodel_6/transformer_encoder_18/Encoder-2nd-NormalizationLayer-1/add/ReadVariableOpRmodel_6/transformer_encoder_18/Encoder-2nd-NormalizationLayer-1/add/ReadVariableOp2�
Rmodel_6/transformer_encoder_18/Encoder-2nd-NormalizationLayer-1/mul/ReadVariableOpRmodel_6/transformer_encoder_18/Encoder-2nd-NormalizationLayer-1/mul/ReadVariableOp2�
Rmodel_6/transformer_encoder_18/Encoder-FeedForwardLayer_1_1/BiasAdd/ReadVariableOpRmodel_6/transformer_encoder_18/Encoder-FeedForwardLayer_1_1/BiasAdd/ReadVariableOp2�
Tmodel_6/transformer_encoder_18/Encoder-FeedForwardLayer_1_1/Tensordot/ReadVariableOpTmodel_6/transformer_encoder_18/Encoder-FeedForwardLayer_1_1/Tensordot/ReadVariableOp2�
Rmodel_6/transformer_encoder_18/Encoder-FeedForwardLayer_2_1/BiasAdd/ReadVariableOpRmodel_6/transformer_encoder_18/Encoder-FeedForwardLayer_2_1/BiasAdd/ReadVariableOp2�
Tmodel_6/transformer_encoder_18/Encoder-FeedForwardLayer_2_1/Tensordot/ReadVariableOpTmodel_6/transformer_encoder_18/Encoder-FeedForwardLayer_2_1/Tensordot/ReadVariableOp2�
_model_6/transformer_encoder_18/Encoder-SelfAttentionLayer-1/attention_output/add/ReadVariableOp_model_6/transformer_encoder_18/Encoder-SelfAttentionLayer-1/attention_output/add/ReadVariableOp2�
imodel_6/transformer_encoder_18/Encoder-SelfAttentionLayer-1/attention_output/einsum/Einsum/ReadVariableOpimodel_6/transformer_encoder_18/Encoder-SelfAttentionLayer-1/attention_output/einsum/Einsum/ReadVariableOp2�
Rmodel_6/transformer_encoder_18/Encoder-SelfAttentionLayer-1/key/add/ReadVariableOpRmodel_6/transformer_encoder_18/Encoder-SelfAttentionLayer-1/key/add/ReadVariableOp2�
\model_6/transformer_encoder_18/Encoder-SelfAttentionLayer-1/key/einsum/Einsum/ReadVariableOp\model_6/transformer_encoder_18/Encoder-SelfAttentionLayer-1/key/einsum/Einsum/ReadVariableOp2�
Tmodel_6/transformer_encoder_18/Encoder-SelfAttentionLayer-1/query/add/ReadVariableOpTmodel_6/transformer_encoder_18/Encoder-SelfAttentionLayer-1/query/add/ReadVariableOp2�
^model_6/transformer_encoder_18/Encoder-SelfAttentionLayer-1/query/einsum/Einsum/ReadVariableOp^model_6/transformer_encoder_18/Encoder-SelfAttentionLayer-1/query/einsum/Einsum/ReadVariableOp2�
Tmodel_6/transformer_encoder_18/Encoder-SelfAttentionLayer-1/value/add/ReadVariableOpTmodel_6/transformer_encoder_18/Encoder-SelfAttentionLayer-1/value/add/ReadVariableOp2�
^model_6/transformer_encoder_18/Encoder-SelfAttentionLayer-1/value/einsum/Einsum/ReadVariableOp^model_6/transformer_encoder_18/Encoder-SelfAttentionLayer-1/value/einsum/Einsum/ReadVariableOp2�
Rmodel_6/transformer_encoder_19/Encoder-1st-NormalizationLayer-2/add/ReadVariableOpRmodel_6/transformer_encoder_19/Encoder-1st-NormalizationLayer-2/add/ReadVariableOp2�
Rmodel_6/transformer_encoder_19/Encoder-1st-NormalizationLayer-2/mul/ReadVariableOpRmodel_6/transformer_encoder_19/Encoder-1st-NormalizationLayer-2/mul/ReadVariableOp2�
Rmodel_6/transformer_encoder_19/Encoder-2nd-NormalizationLayer-2/add/ReadVariableOpRmodel_6/transformer_encoder_19/Encoder-2nd-NormalizationLayer-2/add/ReadVariableOp2�
Rmodel_6/transformer_encoder_19/Encoder-2nd-NormalizationLayer-2/mul/ReadVariableOpRmodel_6/transformer_encoder_19/Encoder-2nd-NormalizationLayer-2/mul/ReadVariableOp2�
Rmodel_6/transformer_encoder_19/Encoder-FeedForwardLayer_1_2/BiasAdd/ReadVariableOpRmodel_6/transformer_encoder_19/Encoder-FeedForwardLayer_1_2/BiasAdd/ReadVariableOp2�
Tmodel_6/transformer_encoder_19/Encoder-FeedForwardLayer_1_2/Tensordot/ReadVariableOpTmodel_6/transformer_encoder_19/Encoder-FeedForwardLayer_1_2/Tensordot/ReadVariableOp2�
Rmodel_6/transformer_encoder_19/Encoder-FeedForwardLayer_2_2/BiasAdd/ReadVariableOpRmodel_6/transformer_encoder_19/Encoder-FeedForwardLayer_2_2/BiasAdd/ReadVariableOp2�
Tmodel_6/transformer_encoder_19/Encoder-FeedForwardLayer_2_2/Tensordot/ReadVariableOpTmodel_6/transformer_encoder_19/Encoder-FeedForwardLayer_2_2/Tensordot/ReadVariableOp2�
_model_6/transformer_encoder_19/Encoder-SelfAttentionLayer-2/attention_output/add/ReadVariableOp_model_6/transformer_encoder_19/Encoder-SelfAttentionLayer-2/attention_output/add/ReadVariableOp2�
imodel_6/transformer_encoder_19/Encoder-SelfAttentionLayer-2/attention_output/einsum/Einsum/ReadVariableOpimodel_6/transformer_encoder_19/Encoder-SelfAttentionLayer-2/attention_output/einsum/Einsum/ReadVariableOp2�
Rmodel_6/transformer_encoder_19/Encoder-SelfAttentionLayer-2/key/add/ReadVariableOpRmodel_6/transformer_encoder_19/Encoder-SelfAttentionLayer-2/key/add/ReadVariableOp2�
\model_6/transformer_encoder_19/Encoder-SelfAttentionLayer-2/key/einsum/Einsum/ReadVariableOp\model_6/transformer_encoder_19/Encoder-SelfAttentionLayer-2/key/einsum/Einsum/ReadVariableOp2�
Tmodel_6/transformer_encoder_19/Encoder-SelfAttentionLayer-2/query/add/ReadVariableOpTmodel_6/transformer_encoder_19/Encoder-SelfAttentionLayer-2/query/add/ReadVariableOp2�
^model_6/transformer_encoder_19/Encoder-SelfAttentionLayer-2/query/einsum/Einsum/ReadVariableOp^model_6/transformer_encoder_19/Encoder-SelfAttentionLayer-2/query/einsum/Einsum/ReadVariableOp2�
Tmodel_6/transformer_encoder_19/Encoder-SelfAttentionLayer-2/value/add/ReadVariableOpTmodel_6/transformer_encoder_19/Encoder-SelfAttentionLayer-2/value/add/ReadVariableOp2�
^model_6/transformer_encoder_19/Encoder-SelfAttentionLayer-2/value/einsum/Einsum/ReadVariableOp^model_6/transformer_encoder_19/Encoder-SelfAttentionLayer-2/value/einsum/Einsum/ReadVariableOp2�
Rmodel_6/transformer_encoder_20/Encoder-1st-NormalizationLayer-3/add/ReadVariableOpRmodel_6/transformer_encoder_20/Encoder-1st-NormalizationLayer-3/add/ReadVariableOp2�
Rmodel_6/transformer_encoder_20/Encoder-1st-NormalizationLayer-3/mul/ReadVariableOpRmodel_6/transformer_encoder_20/Encoder-1st-NormalizationLayer-3/mul/ReadVariableOp2�
Rmodel_6/transformer_encoder_20/Encoder-2nd-NormalizationLayer-3/add/ReadVariableOpRmodel_6/transformer_encoder_20/Encoder-2nd-NormalizationLayer-3/add/ReadVariableOp2�
Rmodel_6/transformer_encoder_20/Encoder-2nd-NormalizationLayer-3/mul/ReadVariableOpRmodel_6/transformer_encoder_20/Encoder-2nd-NormalizationLayer-3/mul/ReadVariableOp2�
Rmodel_6/transformer_encoder_20/Encoder-FeedForwardLayer_1_3/BiasAdd/ReadVariableOpRmodel_6/transformer_encoder_20/Encoder-FeedForwardLayer_1_3/BiasAdd/ReadVariableOp2�
Tmodel_6/transformer_encoder_20/Encoder-FeedForwardLayer_1_3/Tensordot/ReadVariableOpTmodel_6/transformer_encoder_20/Encoder-FeedForwardLayer_1_3/Tensordot/ReadVariableOp2�
Rmodel_6/transformer_encoder_20/Encoder-FeedForwardLayer_2_3/BiasAdd/ReadVariableOpRmodel_6/transformer_encoder_20/Encoder-FeedForwardLayer_2_3/BiasAdd/ReadVariableOp2�
Tmodel_6/transformer_encoder_20/Encoder-FeedForwardLayer_2_3/Tensordot/ReadVariableOpTmodel_6/transformer_encoder_20/Encoder-FeedForwardLayer_2_3/Tensordot/ReadVariableOp2�
_model_6/transformer_encoder_20/Encoder-SelfAttentionLayer-3/attention_output/add/ReadVariableOp_model_6/transformer_encoder_20/Encoder-SelfAttentionLayer-3/attention_output/add/ReadVariableOp2�
imodel_6/transformer_encoder_20/Encoder-SelfAttentionLayer-3/attention_output/einsum/Einsum/ReadVariableOpimodel_6/transformer_encoder_20/Encoder-SelfAttentionLayer-3/attention_output/einsum/Einsum/ReadVariableOp2�
Rmodel_6/transformer_encoder_20/Encoder-SelfAttentionLayer-3/key/add/ReadVariableOpRmodel_6/transformer_encoder_20/Encoder-SelfAttentionLayer-3/key/add/ReadVariableOp2�
\model_6/transformer_encoder_20/Encoder-SelfAttentionLayer-3/key/einsum/Einsum/ReadVariableOp\model_6/transformer_encoder_20/Encoder-SelfAttentionLayer-3/key/einsum/Einsum/ReadVariableOp2�
Tmodel_6/transformer_encoder_20/Encoder-SelfAttentionLayer-3/query/add/ReadVariableOpTmodel_6/transformer_encoder_20/Encoder-SelfAttentionLayer-3/query/add/ReadVariableOp2�
^model_6/transformer_encoder_20/Encoder-SelfAttentionLayer-3/query/einsum/Einsum/ReadVariableOp^model_6/transformer_encoder_20/Encoder-SelfAttentionLayer-3/query/einsum/Einsum/ReadVariableOp2�
Tmodel_6/transformer_encoder_20/Encoder-SelfAttentionLayer-3/value/add/ReadVariableOpTmodel_6/transformer_encoder_20/Encoder-SelfAttentionLayer-3/value/add/ReadVariableOp2�
^model_6/transformer_encoder_20/Encoder-SelfAttentionLayer-3/value/einsum/Einsum/ReadVariableOp^model_6/transformer_encoder_20/Encoder-SelfAttentionLayer-3/value/einsum/Einsum/ReadVariableOp:(7$
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
�
y
]__inference_ReduceStackDimensionViaSummation_layer_call_and_return_conditional_losses_1473628

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
^
2__inference_ConcatenateLayer_layer_call_fn_1475748
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
M__inference_ConcatenateLayer_layer_call_and_return_conditional_losses_1472938`
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
inputs_0"�L
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
)__inference_model_6_layer_call_fn_1473772
)__inference_model_6_layer_call_fn_1473886�
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
D__inference_model_6_layer_call_and_return_conditional_losses_1472986
D__inference_model_6_layer_call_and_return_conditional_losses_1473658�
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
"__inference__wrapped_model_1472183StackLevelInputFeaturesTimeLimitInput"�
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
.__inference_MaskingLayer_layer_call_fn_1474308�
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
I__inference_MaskingLayer_layer_call_and_return_conditional_losses_1474319�
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
8__inference_transformer_encoder_18_layer_call_fn_1474358
8__inference_transformer_encoder_18_layer_call_fn_1474397�
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
S__inference_transformer_encoder_18_layer_call_and_return_conditional_losses_1474585
S__inference_transformer_encoder_18_layer_call_and_return_conditional_losses_1474759�
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
8__inference_transformer_encoder_19_layer_call_fn_1474798
8__inference_transformer_encoder_19_layer_call_fn_1474837�
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
S__inference_transformer_encoder_19_layer_call_and_return_conditional_losses_1475025
S__inference_transformer_encoder_19_layer_call_and_return_conditional_losses_1475199�
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
8__inference_transformer_encoder_20_layer_call_fn_1475238
8__inference_transformer_encoder_20_layer_call_fn_1475277�
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
S__inference_transformer_encoder_20_layer_call_and_return_conditional_losses_1475465
S__inference_transformer_encoder_20_layer_call_and_return_conditional_losses_1475639�
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
0__inference_FinalLayerNorm_layer_call_fn_1475648�
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
K__inference_FinalLayerNorm_layer_call_and_return_conditional_losses_1475690�
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
B__inference_ReduceStackDimensionViaSummation_layer_call_fn_1475695
B__inference_ReduceStackDimensionViaSummation_layer_call_fn_1475700�
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
]__inference_ReduceStackDimensionViaSummation_layer_call_and_return_conditional_losses_1475708
]__inference_ReduceStackDimensionViaSummation_layer_call_and_return_conditional_losses_1475716�
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
6__inference_StandardizeTimeLimit_layer_call_fn_1475721
6__inference_StandardizeTimeLimit_layer_call_fn_1475726�
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
Q__inference_StandardizeTimeLimit_layer_call_and_return_conditional_losses_1475734
Q__inference_StandardizeTimeLimit_layer_call_and_return_conditional_losses_1475742�
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
2__inference_ConcatenateLayer_layer_call_fn_1475748�
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
M__inference_ConcatenateLayer_layer_call_and_return_conditional_losses_1475755�
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
>__inference_FullyConnectedLayerAreaRatio_layer_call_fn_1475764�
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
Y__inference_FullyConnectedLayerAreaRatio_layer_call_and_return_conditional_losses_1475782�
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
5__inference_PredictionAreaRatio_layer_call_fn_1475791�
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
P__inference_PredictionAreaRatio_layer_call_and_return_conditional_losses_1475802�
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
(__inference_Output_layer_call_fn_1475807
(__inference_Output_layer_call_fn_1475812�
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
C__inference_Output_layer_call_and_return_conditional_losses_1475817
C__inference_Output_layer_call_and_return_conditional_losses_1475822�
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
V:T	2@transformer_encoder_18/Encoder-SelfAttentionLayer-1/query/kernel
P:N2>transformer_encoder_18/Encoder-SelfAttentionLayer-1/query/bias
T:R	2>transformer_encoder_18/Encoder-SelfAttentionLayer-1/key/kernel
N:L2<transformer_encoder_18/Encoder-SelfAttentionLayer-1/key/bias
V:T	2@transformer_encoder_18/Encoder-SelfAttentionLayer-1/value/kernel
P:N2>transformer_encoder_18/Encoder-SelfAttentionLayer-1/value/bias
a:_	2Ktransformer_encoder_18/Encoder-SelfAttentionLayer-1/attention_output/kernel
W:U	2Itransformer_encoder_18/Encoder-SelfAttentionLayer-1/attention_output/bias
K:I	2=transformer_encoder_18/Encoder-1st-NormalizationLayer-1/gamma
J:H	2<transformer_encoder_18/Encoder-1st-NormalizationLayer-1/beta
K:I	2=transformer_encoder_18/Encoder-2nd-NormalizationLayer-1/gamma
J:H	2<transformer_encoder_18/Encoder-2nd-NormalizationLayer-1/beta
L:J		2:transformer_encoder_18/Encoder-FeedForwardLayer_1_1/kernel
F:D	28transformer_encoder_18/Encoder-FeedForwardLayer_1_1/bias
L:J		2:transformer_encoder_18/Encoder-FeedForwardLayer_2_1/kernel
F:D	28transformer_encoder_18/Encoder-FeedForwardLayer_2_1/bias
V:T	2@transformer_encoder_19/Encoder-SelfAttentionLayer-2/query/kernel
P:N2>transformer_encoder_19/Encoder-SelfAttentionLayer-2/query/bias
T:R	2>transformer_encoder_19/Encoder-SelfAttentionLayer-2/key/kernel
N:L2<transformer_encoder_19/Encoder-SelfAttentionLayer-2/key/bias
V:T	2@transformer_encoder_19/Encoder-SelfAttentionLayer-2/value/kernel
P:N2>transformer_encoder_19/Encoder-SelfAttentionLayer-2/value/bias
a:_	2Ktransformer_encoder_19/Encoder-SelfAttentionLayer-2/attention_output/kernel
W:U	2Itransformer_encoder_19/Encoder-SelfAttentionLayer-2/attention_output/bias
K:I	2=transformer_encoder_19/Encoder-1st-NormalizationLayer-2/gamma
J:H	2<transformer_encoder_19/Encoder-1st-NormalizationLayer-2/beta
K:I	2=transformer_encoder_19/Encoder-2nd-NormalizationLayer-2/gamma
J:H	2<transformer_encoder_19/Encoder-2nd-NormalizationLayer-2/beta
L:J		2:transformer_encoder_19/Encoder-FeedForwardLayer_1_2/kernel
F:D	28transformer_encoder_19/Encoder-FeedForwardLayer_1_2/bias
L:J		2:transformer_encoder_19/Encoder-FeedForwardLayer_2_2/kernel
F:D	28transformer_encoder_19/Encoder-FeedForwardLayer_2_2/bias
V:T	2@transformer_encoder_20/Encoder-SelfAttentionLayer-3/query/kernel
P:N2>transformer_encoder_20/Encoder-SelfAttentionLayer-3/query/bias
T:R	2>transformer_encoder_20/Encoder-SelfAttentionLayer-3/key/kernel
N:L2<transformer_encoder_20/Encoder-SelfAttentionLayer-3/key/bias
V:T	2@transformer_encoder_20/Encoder-SelfAttentionLayer-3/value/kernel
P:N2>transformer_encoder_20/Encoder-SelfAttentionLayer-3/value/bias
a:_	2Ktransformer_encoder_20/Encoder-SelfAttentionLayer-3/attention_output/kernel
W:U	2Itransformer_encoder_20/Encoder-SelfAttentionLayer-3/attention_output/bias
K:I	2=transformer_encoder_20/Encoder-1st-NormalizationLayer-3/gamma
J:H	2<transformer_encoder_20/Encoder-1st-NormalizationLayer-3/beta
K:I	2=transformer_encoder_20/Encoder-2nd-NormalizationLayer-3/gamma
J:H	2<transformer_encoder_20/Encoder-2nd-NormalizationLayer-3/beta
L:J		2:transformer_encoder_20/Encoder-FeedForwardLayer_1_3/kernel
F:D	28transformer_encoder_20/Encoder-FeedForwardLayer_1_3/bias
L:J		2:transformer_encoder_20/Encoder-FeedForwardLayer_2_3/kernel
F:D	28transformer_encoder_20/Encoder-FeedForwardLayer_2_3/bias
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
)__inference_model_6_layer_call_fn_1473772StackLevelInputFeaturesTimeLimitInput"�
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
)__inference_model_6_layer_call_fn_1473886StackLevelInputFeaturesTimeLimitInput"�
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
D__inference_model_6_layer_call_and_return_conditional_losses_1472986StackLevelInputFeaturesTimeLimitInput"�
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
D__inference_model_6_layer_call_and_return_conditional_losses_1473658StackLevelInputFeaturesTimeLimitInput"�
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
%__inference_signature_wrapper_1474303StackLevelInputFeaturesTimeLimitInput"�
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
.__inference_MaskingLayer_layer_call_fn_1474308inputs"�
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
I__inference_MaskingLayer_layer_call_and_return_conditional_losses_1474319inputs"�
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
8__inference_transformer_encoder_18_layer_call_fn_1474358inputs"�
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
8__inference_transformer_encoder_18_layer_call_fn_1474397inputs"�
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
S__inference_transformer_encoder_18_layer_call_and_return_conditional_losses_1474585inputs"�
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
S__inference_transformer_encoder_18_layer_call_and_return_conditional_losses_1474759inputs"�
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
8__inference_transformer_encoder_19_layer_call_fn_1474798inputs"�
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
8__inference_transformer_encoder_19_layer_call_fn_1474837inputs"�
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
S__inference_transformer_encoder_19_layer_call_and_return_conditional_losses_1475025inputs"�
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
S__inference_transformer_encoder_19_layer_call_and_return_conditional_losses_1475199inputs"�
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
8__inference_transformer_encoder_20_layer_call_fn_1475238inputs"�
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
8__inference_transformer_encoder_20_layer_call_fn_1475277inputs"�
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
S__inference_transformer_encoder_20_layer_call_and_return_conditional_losses_1475465inputs"�
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
S__inference_transformer_encoder_20_layer_call_and_return_conditional_losses_1475639inputs"�
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
0__inference_FinalLayerNorm_layer_call_fn_1475648inputs"�
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
K__inference_FinalLayerNorm_layer_call_and_return_conditional_losses_1475690inputs"�
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
B__inference_ReduceStackDimensionViaSummation_layer_call_fn_1475695inputs"�
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
B__inference_ReduceStackDimensionViaSummation_layer_call_fn_1475700inputs"�
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
]__inference_ReduceStackDimensionViaSummation_layer_call_and_return_conditional_losses_1475708inputs"�
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
]__inference_ReduceStackDimensionViaSummation_layer_call_and_return_conditional_losses_1475716inputs"�
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
6__inference_StandardizeTimeLimit_layer_call_fn_1475721inputs"�
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
6__inference_StandardizeTimeLimit_layer_call_fn_1475726inputs"�
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
Q__inference_StandardizeTimeLimit_layer_call_and_return_conditional_losses_1475734inputs"�
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
Q__inference_StandardizeTimeLimit_layer_call_and_return_conditional_losses_1475742inputs"�
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
2__inference_ConcatenateLayer_layer_call_fn_1475748inputs_0inputs_1"�
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
M__inference_ConcatenateLayer_layer_call_and_return_conditional_losses_1475755inputs_0inputs_1"�
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
>__inference_FullyConnectedLayerAreaRatio_layer_call_fn_1475764inputs"�
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
Y__inference_FullyConnectedLayerAreaRatio_layer_call_and_return_conditional_losses_1475782inputs"�
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
5__inference_PredictionAreaRatio_layer_call_fn_1475791inputs"�
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
P__inference_PredictionAreaRatio_layer_call_and_return_conditional_losses_1475802inputs"�
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
(__inference_Output_layer_call_fn_1475807inputs"�
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
(__inference_Output_layer_call_fn_1475812inputs"�
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
C__inference_Output_layer_call_and_return_conditional_losses_1475817inputs"�
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
C__inference_Output_layer_call_and_return_conditional_losses_1475822inputs"�
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
M__inference_ConcatenateLayer_layer_call_and_return_conditional_losses_1475755�Z�W
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
2__inference_ConcatenateLayer_layer_call_fn_1475748Z�W
P�M
K�H
"�
inputs_0���������	
"�
inputs_1���������
� "!�
unknown���������
�
K__inference_FinalLayerNorm_layer_call_and_return_conditional_losses_1475690kMN3�0
)�&
$�!
inputs���������P	
� "0�-
&�#
tensor_0���������P	
� �
0__inference_FinalLayerNorm_layer_call_fn_1475648`MN3�0
)�&
$�!
inputs���������P	
� "%�"
unknown���������P	�
Y__inference_FullyConnectedLayerAreaRatio_layer_call_and_return_conditional_losses_1475782cgh/�,
%�"
 �
inputs���������

� ",�)
"�
tensor_0���������

� �
>__inference_FullyConnectedLayerAreaRatio_layer_call_fn_1475764Xgh/�,
%�"
 �
inputs���������

� "!�
unknown���������
�
I__inference_MaskingLayer_layer_call_and_return_conditional_losses_1474319g3�0
)�&
$�!
inputs���������P	
� "0�-
&�#
tensor_0���������P	
� �
.__inference_MaskingLayer_layer_call_fn_1474308\3�0
)�&
$�!
inputs���������P	
� "%�"
unknown���������P	�
C__inference_Output_layer_call_and_return_conditional_losses_1475817X7�4
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
C__inference_Output_layer_call_and_return_conditional_losses_1475822X7�4
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
(__inference_Output_layer_call_fn_1475807M7�4
-�*
 �
inputs���������

 
p
� "�
unknowny
(__inference_Output_layer_call_fn_1475812M7�4
-�*
 �
inputs���������

 
p 
� "�
unknown�
P__inference_PredictionAreaRatio_layer_call_and_return_conditional_losses_1475802cop/�,
%�"
 �
inputs���������

� ",�)
"�
tensor_0���������
� �
5__inference_PredictionAreaRatio_layer_call_fn_1475791Xop/�,
%�"
 �
inputs���������

� "!�
unknown����������
]__inference_ReduceStackDimensionViaSummation_layer_call_and_return_conditional_losses_1475708k;�8
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
]__inference_ReduceStackDimensionViaSummation_layer_call_and_return_conditional_losses_1475716k;�8
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
B__inference_ReduceStackDimensionViaSummation_layer_call_fn_1475695`;�8
1�.
$�!
inputs���������P	

 
p
� "!�
unknown���������	�
B__inference_ReduceStackDimensionViaSummation_layer_call_fn_1475700`;�8
1�.
$�!
inputs���������P	

 
p 
� "!�
unknown���������	�
Q__inference_StandardizeTimeLimit_layer_call_and_return_conditional_losses_1475734g7�4
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
Q__inference_StandardizeTimeLimit_layer_call_and_return_conditional_losses_1475742g7�4
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
6__inference_StandardizeTimeLimit_layer_call_fn_1475721\7�4
-�*
 �
inputs���������

 
p
� "!�
unknown����������
6__inference_StandardizeTimeLimit_layer_call_fn_1475726\7�4
-�*
 �
inputs���������

 
p 
� "!�
unknown����������
"__inference__wrapped_model_1472183�]�wxyz{|}~��������������������������������������MNghops�p
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
D__inference_model_6_layer_call_and_return_conditional_losses_1472986�]�wxyz{|}~��������������������������������������MNghop{�x
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
D__inference_model_6_layer_call_and_return_conditional_losses_1473658�]�wxyz{|}~��������������������������������������MNghop{�x
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
)__inference_model_6_layer_call_fn_1473772�]�wxyz{|}~��������������������������������������MNghop{�x
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
)__inference_model_6_layer_call_fn_1473886�]�wxyz{|}~��������������������������������������MNghop{�x
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
%__inference_signature_wrapper_1474303�]�wxyz{|}~��������������������������������������MNghop���
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
S__inference_transformer_encoder_18_layer_call_and_return_conditional_losses_1474585��wxyz{|}~������C�@
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
S__inference_transformer_encoder_18_layer_call_and_return_conditional_losses_1474759��wxyz{|}~������C�@
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
8__inference_transformer_encoder_18_layer_call_fn_1474358��wxyz{|}~������C�@
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
8__inference_transformer_encoder_18_layer_call_fn_1474397��wxyz{|}~������C�@
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
S__inference_transformer_encoder_19_layer_call_and_return_conditional_losses_1475025� ����������������C�@
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
S__inference_transformer_encoder_19_layer_call_and_return_conditional_losses_1475199� ����������������C�@
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
8__inference_transformer_encoder_19_layer_call_fn_1474798� ����������������C�@
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
8__inference_transformer_encoder_19_layer_call_fn_1474837� ����������������C�@
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
S__inference_transformer_encoder_20_layer_call_and_return_conditional_losses_1475465� ����������������C�@
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
S__inference_transformer_encoder_20_layer_call_and_return_conditional_losses_1475639� ����������������C�@
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
8__inference_transformer_encoder_20_layer_call_fn_1475238� ����������������C�@
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
8__inference_transformer_encoder_20_layer_call_fn_1475277� ����������������C�@
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