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
8transformer_encoder_29/Encoder-FeedForwardLayer_2_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*I
shared_name:8transformer_encoder_29/Encoder-FeedForwardLayer_2_3/bias
�
Ltransformer_encoder_29/Encoder-FeedForwardLayer_2_3/bias/Read/ReadVariableOpReadVariableOp8transformer_encoder_29/Encoder-FeedForwardLayer_2_3/bias*
_output_shapes
:	*
dtype0
�
:transformer_encoder_29/Encoder-FeedForwardLayer_2_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:		*K
shared_name<:transformer_encoder_29/Encoder-FeedForwardLayer_2_3/kernel
�
Ntransformer_encoder_29/Encoder-FeedForwardLayer_2_3/kernel/Read/ReadVariableOpReadVariableOp:transformer_encoder_29/Encoder-FeedForwardLayer_2_3/kernel*
_output_shapes

:		*
dtype0
�
8transformer_encoder_29/Encoder-FeedForwardLayer_1_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*I
shared_name:8transformer_encoder_29/Encoder-FeedForwardLayer_1_3/bias
�
Ltransformer_encoder_29/Encoder-FeedForwardLayer_1_3/bias/Read/ReadVariableOpReadVariableOp8transformer_encoder_29/Encoder-FeedForwardLayer_1_3/bias*
_output_shapes
:	*
dtype0
�
:transformer_encoder_29/Encoder-FeedForwardLayer_1_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:		*K
shared_name<:transformer_encoder_29/Encoder-FeedForwardLayer_1_3/kernel
�
Ntransformer_encoder_29/Encoder-FeedForwardLayer_1_3/kernel/Read/ReadVariableOpReadVariableOp:transformer_encoder_29/Encoder-FeedForwardLayer_1_3/kernel*
_output_shapes

:		*
dtype0
�
<transformer_encoder_29/Encoder-2nd-NormalizationLayer-3/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*M
shared_name><transformer_encoder_29/Encoder-2nd-NormalizationLayer-3/beta
�
Ptransformer_encoder_29/Encoder-2nd-NormalizationLayer-3/beta/Read/ReadVariableOpReadVariableOp<transformer_encoder_29/Encoder-2nd-NormalizationLayer-3/beta*
_output_shapes
:	*
dtype0
�
=transformer_encoder_29/Encoder-2nd-NormalizationLayer-3/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*N
shared_name?=transformer_encoder_29/Encoder-2nd-NormalizationLayer-3/gamma
�
Qtransformer_encoder_29/Encoder-2nd-NormalizationLayer-3/gamma/Read/ReadVariableOpReadVariableOp=transformer_encoder_29/Encoder-2nd-NormalizationLayer-3/gamma*
_output_shapes
:	*
dtype0
�
<transformer_encoder_29/Encoder-1st-NormalizationLayer-3/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*M
shared_name><transformer_encoder_29/Encoder-1st-NormalizationLayer-3/beta
�
Ptransformer_encoder_29/Encoder-1st-NormalizationLayer-3/beta/Read/ReadVariableOpReadVariableOp<transformer_encoder_29/Encoder-1st-NormalizationLayer-3/beta*
_output_shapes
:	*
dtype0
�
=transformer_encoder_29/Encoder-1st-NormalizationLayer-3/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*N
shared_name?=transformer_encoder_29/Encoder-1st-NormalizationLayer-3/gamma
�
Qtransformer_encoder_29/Encoder-1st-NormalizationLayer-3/gamma/Read/ReadVariableOpReadVariableOp=transformer_encoder_29/Encoder-1st-NormalizationLayer-3/gamma*
_output_shapes
:	*
dtype0
�
Itransformer_encoder_29/Encoder-SelfAttentionLayer-3/attention_output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*Z
shared_nameKItransformer_encoder_29/Encoder-SelfAttentionLayer-3/attention_output/bias
�
]transformer_encoder_29/Encoder-SelfAttentionLayer-3/attention_output/bias/Read/ReadVariableOpReadVariableOpItransformer_encoder_29/Encoder-SelfAttentionLayer-3/attention_output/bias*
_output_shapes
:	*
dtype0
�
Ktransformer_encoder_29/Encoder-SelfAttentionLayer-3/attention_output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*\
shared_nameMKtransformer_encoder_29/Encoder-SelfAttentionLayer-3/attention_output/kernel
�
_transformer_encoder_29/Encoder-SelfAttentionLayer-3/attention_output/kernel/Read/ReadVariableOpReadVariableOpKtransformer_encoder_29/Encoder-SelfAttentionLayer-3/attention_output/kernel*"
_output_shapes
:	*
dtype0
�
>transformer_encoder_29/Encoder-SelfAttentionLayer-3/value/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*O
shared_name@>transformer_encoder_29/Encoder-SelfAttentionLayer-3/value/bias
�
Rtransformer_encoder_29/Encoder-SelfAttentionLayer-3/value/bias/Read/ReadVariableOpReadVariableOp>transformer_encoder_29/Encoder-SelfAttentionLayer-3/value/bias*
_output_shapes

:*
dtype0
�
@transformer_encoder_29/Encoder-SelfAttentionLayer-3/value/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*Q
shared_nameB@transformer_encoder_29/Encoder-SelfAttentionLayer-3/value/kernel
�
Ttransformer_encoder_29/Encoder-SelfAttentionLayer-3/value/kernel/Read/ReadVariableOpReadVariableOp@transformer_encoder_29/Encoder-SelfAttentionLayer-3/value/kernel*"
_output_shapes
:	*
dtype0
�
<transformer_encoder_29/Encoder-SelfAttentionLayer-3/key/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*M
shared_name><transformer_encoder_29/Encoder-SelfAttentionLayer-3/key/bias
�
Ptransformer_encoder_29/Encoder-SelfAttentionLayer-3/key/bias/Read/ReadVariableOpReadVariableOp<transformer_encoder_29/Encoder-SelfAttentionLayer-3/key/bias*
_output_shapes

:*
dtype0
�
>transformer_encoder_29/Encoder-SelfAttentionLayer-3/key/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*O
shared_name@>transformer_encoder_29/Encoder-SelfAttentionLayer-3/key/kernel
�
Rtransformer_encoder_29/Encoder-SelfAttentionLayer-3/key/kernel/Read/ReadVariableOpReadVariableOp>transformer_encoder_29/Encoder-SelfAttentionLayer-3/key/kernel*"
_output_shapes
:	*
dtype0
�
>transformer_encoder_29/Encoder-SelfAttentionLayer-3/query/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*O
shared_name@>transformer_encoder_29/Encoder-SelfAttentionLayer-3/query/bias
�
Rtransformer_encoder_29/Encoder-SelfAttentionLayer-3/query/bias/Read/ReadVariableOpReadVariableOp>transformer_encoder_29/Encoder-SelfAttentionLayer-3/query/bias*
_output_shapes

:*
dtype0
�
@transformer_encoder_29/Encoder-SelfAttentionLayer-3/query/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*Q
shared_nameB@transformer_encoder_29/Encoder-SelfAttentionLayer-3/query/kernel
�
Ttransformer_encoder_29/Encoder-SelfAttentionLayer-3/query/kernel/Read/ReadVariableOpReadVariableOp@transformer_encoder_29/Encoder-SelfAttentionLayer-3/query/kernel*"
_output_shapes
:	*
dtype0
�
8transformer_encoder_28/Encoder-FeedForwardLayer_2_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*I
shared_name:8transformer_encoder_28/Encoder-FeedForwardLayer_2_2/bias
�
Ltransformer_encoder_28/Encoder-FeedForwardLayer_2_2/bias/Read/ReadVariableOpReadVariableOp8transformer_encoder_28/Encoder-FeedForwardLayer_2_2/bias*
_output_shapes
:	*
dtype0
�
:transformer_encoder_28/Encoder-FeedForwardLayer_2_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:		*K
shared_name<:transformer_encoder_28/Encoder-FeedForwardLayer_2_2/kernel
�
Ntransformer_encoder_28/Encoder-FeedForwardLayer_2_2/kernel/Read/ReadVariableOpReadVariableOp:transformer_encoder_28/Encoder-FeedForwardLayer_2_2/kernel*
_output_shapes

:		*
dtype0
�
8transformer_encoder_28/Encoder-FeedForwardLayer_1_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*I
shared_name:8transformer_encoder_28/Encoder-FeedForwardLayer_1_2/bias
�
Ltransformer_encoder_28/Encoder-FeedForwardLayer_1_2/bias/Read/ReadVariableOpReadVariableOp8transformer_encoder_28/Encoder-FeedForwardLayer_1_2/bias*
_output_shapes
:	*
dtype0
�
:transformer_encoder_28/Encoder-FeedForwardLayer_1_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:		*K
shared_name<:transformer_encoder_28/Encoder-FeedForwardLayer_1_2/kernel
�
Ntransformer_encoder_28/Encoder-FeedForwardLayer_1_2/kernel/Read/ReadVariableOpReadVariableOp:transformer_encoder_28/Encoder-FeedForwardLayer_1_2/kernel*
_output_shapes

:		*
dtype0
�
<transformer_encoder_28/Encoder-2nd-NormalizationLayer-2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*M
shared_name><transformer_encoder_28/Encoder-2nd-NormalizationLayer-2/beta
�
Ptransformer_encoder_28/Encoder-2nd-NormalizationLayer-2/beta/Read/ReadVariableOpReadVariableOp<transformer_encoder_28/Encoder-2nd-NormalizationLayer-2/beta*
_output_shapes
:	*
dtype0
�
=transformer_encoder_28/Encoder-2nd-NormalizationLayer-2/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*N
shared_name?=transformer_encoder_28/Encoder-2nd-NormalizationLayer-2/gamma
�
Qtransformer_encoder_28/Encoder-2nd-NormalizationLayer-2/gamma/Read/ReadVariableOpReadVariableOp=transformer_encoder_28/Encoder-2nd-NormalizationLayer-2/gamma*
_output_shapes
:	*
dtype0
�
<transformer_encoder_28/Encoder-1st-NormalizationLayer-2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*M
shared_name><transformer_encoder_28/Encoder-1st-NormalizationLayer-2/beta
�
Ptransformer_encoder_28/Encoder-1st-NormalizationLayer-2/beta/Read/ReadVariableOpReadVariableOp<transformer_encoder_28/Encoder-1st-NormalizationLayer-2/beta*
_output_shapes
:	*
dtype0
�
=transformer_encoder_28/Encoder-1st-NormalizationLayer-2/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*N
shared_name?=transformer_encoder_28/Encoder-1st-NormalizationLayer-2/gamma
�
Qtransformer_encoder_28/Encoder-1st-NormalizationLayer-2/gamma/Read/ReadVariableOpReadVariableOp=transformer_encoder_28/Encoder-1st-NormalizationLayer-2/gamma*
_output_shapes
:	*
dtype0
�
Itransformer_encoder_28/Encoder-SelfAttentionLayer-2/attention_output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*Z
shared_nameKItransformer_encoder_28/Encoder-SelfAttentionLayer-2/attention_output/bias
�
]transformer_encoder_28/Encoder-SelfAttentionLayer-2/attention_output/bias/Read/ReadVariableOpReadVariableOpItransformer_encoder_28/Encoder-SelfAttentionLayer-2/attention_output/bias*
_output_shapes
:	*
dtype0
�
Ktransformer_encoder_28/Encoder-SelfAttentionLayer-2/attention_output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*\
shared_nameMKtransformer_encoder_28/Encoder-SelfAttentionLayer-2/attention_output/kernel
�
_transformer_encoder_28/Encoder-SelfAttentionLayer-2/attention_output/kernel/Read/ReadVariableOpReadVariableOpKtransformer_encoder_28/Encoder-SelfAttentionLayer-2/attention_output/kernel*"
_output_shapes
:	*
dtype0
�
>transformer_encoder_28/Encoder-SelfAttentionLayer-2/value/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*O
shared_name@>transformer_encoder_28/Encoder-SelfAttentionLayer-2/value/bias
�
Rtransformer_encoder_28/Encoder-SelfAttentionLayer-2/value/bias/Read/ReadVariableOpReadVariableOp>transformer_encoder_28/Encoder-SelfAttentionLayer-2/value/bias*
_output_shapes

:*
dtype0
�
@transformer_encoder_28/Encoder-SelfAttentionLayer-2/value/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*Q
shared_nameB@transformer_encoder_28/Encoder-SelfAttentionLayer-2/value/kernel
�
Ttransformer_encoder_28/Encoder-SelfAttentionLayer-2/value/kernel/Read/ReadVariableOpReadVariableOp@transformer_encoder_28/Encoder-SelfAttentionLayer-2/value/kernel*"
_output_shapes
:	*
dtype0
�
<transformer_encoder_28/Encoder-SelfAttentionLayer-2/key/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*M
shared_name><transformer_encoder_28/Encoder-SelfAttentionLayer-2/key/bias
�
Ptransformer_encoder_28/Encoder-SelfAttentionLayer-2/key/bias/Read/ReadVariableOpReadVariableOp<transformer_encoder_28/Encoder-SelfAttentionLayer-2/key/bias*
_output_shapes

:*
dtype0
�
>transformer_encoder_28/Encoder-SelfAttentionLayer-2/key/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*O
shared_name@>transformer_encoder_28/Encoder-SelfAttentionLayer-2/key/kernel
�
Rtransformer_encoder_28/Encoder-SelfAttentionLayer-2/key/kernel/Read/ReadVariableOpReadVariableOp>transformer_encoder_28/Encoder-SelfAttentionLayer-2/key/kernel*"
_output_shapes
:	*
dtype0
�
>transformer_encoder_28/Encoder-SelfAttentionLayer-2/query/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*O
shared_name@>transformer_encoder_28/Encoder-SelfAttentionLayer-2/query/bias
�
Rtransformer_encoder_28/Encoder-SelfAttentionLayer-2/query/bias/Read/ReadVariableOpReadVariableOp>transformer_encoder_28/Encoder-SelfAttentionLayer-2/query/bias*
_output_shapes

:*
dtype0
�
@transformer_encoder_28/Encoder-SelfAttentionLayer-2/query/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*Q
shared_nameB@transformer_encoder_28/Encoder-SelfAttentionLayer-2/query/kernel
�
Ttransformer_encoder_28/Encoder-SelfAttentionLayer-2/query/kernel/Read/ReadVariableOpReadVariableOp@transformer_encoder_28/Encoder-SelfAttentionLayer-2/query/kernel*"
_output_shapes
:	*
dtype0
�
8transformer_encoder_27/Encoder-FeedForwardLayer_2_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*I
shared_name:8transformer_encoder_27/Encoder-FeedForwardLayer_2_1/bias
�
Ltransformer_encoder_27/Encoder-FeedForwardLayer_2_1/bias/Read/ReadVariableOpReadVariableOp8transformer_encoder_27/Encoder-FeedForwardLayer_2_1/bias*
_output_shapes
:	*
dtype0
�
:transformer_encoder_27/Encoder-FeedForwardLayer_2_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:		*K
shared_name<:transformer_encoder_27/Encoder-FeedForwardLayer_2_1/kernel
�
Ntransformer_encoder_27/Encoder-FeedForwardLayer_2_1/kernel/Read/ReadVariableOpReadVariableOp:transformer_encoder_27/Encoder-FeedForwardLayer_2_1/kernel*
_output_shapes

:		*
dtype0
�
8transformer_encoder_27/Encoder-FeedForwardLayer_1_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*I
shared_name:8transformer_encoder_27/Encoder-FeedForwardLayer_1_1/bias
�
Ltransformer_encoder_27/Encoder-FeedForwardLayer_1_1/bias/Read/ReadVariableOpReadVariableOp8transformer_encoder_27/Encoder-FeedForwardLayer_1_1/bias*
_output_shapes
:	*
dtype0
�
:transformer_encoder_27/Encoder-FeedForwardLayer_1_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:		*K
shared_name<:transformer_encoder_27/Encoder-FeedForwardLayer_1_1/kernel
�
Ntransformer_encoder_27/Encoder-FeedForwardLayer_1_1/kernel/Read/ReadVariableOpReadVariableOp:transformer_encoder_27/Encoder-FeedForwardLayer_1_1/kernel*
_output_shapes

:		*
dtype0
�
<transformer_encoder_27/Encoder-2nd-NormalizationLayer-1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*M
shared_name><transformer_encoder_27/Encoder-2nd-NormalizationLayer-1/beta
�
Ptransformer_encoder_27/Encoder-2nd-NormalizationLayer-1/beta/Read/ReadVariableOpReadVariableOp<transformer_encoder_27/Encoder-2nd-NormalizationLayer-1/beta*
_output_shapes
:	*
dtype0
�
=transformer_encoder_27/Encoder-2nd-NormalizationLayer-1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*N
shared_name?=transformer_encoder_27/Encoder-2nd-NormalizationLayer-1/gamma
�
Qtransformer_encoder_27/Encoder-2nd-NormalizationLayer-1/gamma/Read/ReadVariableOpReadVariableOp=transformer_encoder_27/Encoder-2nd-NormalizationLayer-1/gamma*
_output_shapes
:	*
dtype0
�
<transformer_encoder_27/Encoder-1st-NormalizationLayer-1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*M
shared_name><transformer_encoder_27/Encoder-1st-NormalizationLayer-1/beta
�
Ptransformer_encoder_27/Encoder-1st-NormalizationLayer-1/beta/Read/ReadVariableOpReadVariableOp<transformer_encoder_27/Encoder-1st-NormalizationLayer-1/beta*
_output_shapes
:	*
dtype0
�
=transformer_encoder_27/Encoder-1st-NormalizationLayer-1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*N
shared_name?=transformer_encoder_27/Encoder-1st-NormalizationLayer-1/gamma
�
Qtransformer_encoder_27/Encoder-1st-NormalizationLayer-1/gamma/Read/ReadVariableOpReadVariableOp=transformer_encoder_27/Encoder-1st-NormalizationLayer-1/gamma*
_output_shapes
:	*
dtype0
�
Itransformer_encoder_27/Encoder-SelfAttentionLayer-1/attention_output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*Z
shared_nameKItransformer_encoder_27/Encoder-SelfAttentionLayer-1/attention_output/bias
�
]transformer_encoder_27/Encoder-SelfAttentionLayer-1/attention_output/bias/Read/ReadVariableOpReadVariableOpItransformer_encoder_27/Encoder-SelfAttentionLayer-1/attention_output/bias*
_output_shapes
:	*
dtype0
�
Ktransformer_encoder_27/Encoder-SelfAttentionLayer-1/attention_output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*\
shared_nameMKtransformer_encoder_27/Encoder-SelfAttentionLayer-1/attention_output/kernel
�
_transformer_encoder_27/Encoder-SelfAttentionLayer-1/attention_output/kernel/Read/ReadVariableOpReadVariableOpKtransformer_encoder_27/Encoder-SelfAttentionLayer-1/attention_output/kernel*"
_output_shapes
:	*
dtype0
�
>transformer_encoder_27/Encoder-SelfAttentionLayer-1/value/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*O
shared_name@>transformer_encoder_27/Encoder-SelfAttentionLayer-1/value/bias
�
Rtransformer_encoder_27/Encoder-SelfAttentionLayer-1/value/bias/Read/ReadVariableOpReadVariableOp>transformer_encoder_27/Encoder-SelfAttentionLayer-1/value/bias*
_output_shapes

:*
dtype0
�
@transformer_encoder_27/Encoder-SelfAttentionLayer-1/value/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*Q
shared_nameB@transformer_encoder_27/Encoder-SelfAttentionLayer-1/value/kernel
�
Ttransformer_encoder_27/Encoder-SelfAttentionLayer-1/value/kernel/Read/ReadVariableOpReadVariableOp@transformer_encoder_27/Encoder-SelfAttentionLayer-1/value/kernel*"
_output_shapes
:	*
dtype0
�
<transformer_encoder_27/Encoder-SelfAttentionLayer-1/key/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*M
shared_name><transformer_encoder_27/Encoder-SelfAttentionLayer-1/key/bias
�
Ptransformer_encoder_27/Encoder-SelfAttentionLayer-1/key/bias/Read/ReadVariableOpReadVariableOp<transformer_encoder_27/Encoder-SelfAttentionLayer-1/key/bias*
_output_shapes

:*
dtype0
�
>transformer_encoder_27/Encoder-SelfAttentionLayer-1/key/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*O
shared_name@>transformer_encoder_27/Encoder-SelfAttentionLayer-1/key/kernel
�
Rtransformer_encoder_27/Encoder-SelfAttentionLayer-1/key/kernel/Read/ReadVariableOpReadVariableOp>transformer_encoder_27/Encoder-SelfAttentionLayer-1/key/kernel*"
_output_shapes
:	*
dtype0
�
>transformer_encoder_27/Encoder-SelfAttentionLayer-1/query/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*O
shared_name@>transformer_encoder_27/Encoder-SelfAttentionLayer-1/query/bias
�
Rtransformer_encoder_27/Encoder-SelfAttentionLayer-1/query/bias/Read/ReadVariableOpReadVariableOp>transformer_encoder_27/Encoder-SelfAttentionLayer-1/query/bias*
_output_shapes

:*
dtype0
�
@transformer_encoder_27/Encoder-SelfAttentionLayer-1/query/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*Q
shared_nameB@transformer_encoder_27/Encoder-SelfAttentionLayer-1/query/kernel
�
Ttransformer_encoder_27/Encoder-SelfAttentionLayer-1/query/kernel/Read/ReadVariableOpReadVariableOp@transformer_encoder_27/Encoder-SelfAttentionLayer-1/query/kernel*"
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
PredictionStacks/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_namePredictionStacks/bias
{
)PredictionStacks/bias/Read/ReadVariableOpReadVariableOpPredictionStacks/bias*
_output_shapes
:*
dtype0
�
PredictionStacks/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:	*(
shared_namePredictionStacks/kernel
�
+PredictionStacks/kernel/Read/ReadVariableOpReadVariableOpPredictionStacks/kernel*
_output_shapes

:	*
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
StatefulPartitionedCallStatefulPartitionedCall'serving_default_StackLevelInputFeaturesserving_default_TimeLimitInput=transformer_encoder_27/Encoder-1st-NormalizationLayer-1/gamma<transformer_encoder_27/Encoder-1st-NormalizationLayer-1/beta@transformer_encoder_27/Encoder-SelfAttentionLayer-1/query/kernel>transformer_encoder_27/Encoder-SelfAttentionLayer-1/query/bias>transformer_encoder_27/Encoder-SelfAttentionLayer-1/key/kernel<transformer_encoder_27/Encoder-SelfAttentionLayer-1/key/bias@transformer_encoder_27/Encoder-SelfAttentionLayer-1/value/kernel>transformer_encoder_27/Encoder-SelfAttentionLayer-1/value/biasKtransformer_encoder_27/Encoder-SelfAttentionLayer-1/attention_output/kernelItransformer_encoder_27/Encoder-SelfAttentionLayer-1/attention_output/bias=transformer_encoder_27/Encoder-2nd-NormalizationLayer-1/gamma<transformer_encoder_27/Encoder-2nd-NormalizationLayer-1/beta:transformer_encoder_27/Encoder-FeedForwardLayer_1_1/kernel8transformer_encoder_27/Encoder-FeedForwardLayer_1_1/bias:transformer_encoder_27/Encoder-FeedForwardLayer_2_1/kernel8transformer_encoder_27/Encoder-FeedForwardLayer_2_1/bias=transformer_encoder_28/Encoder-1st-NormalizationLayer-2/gamma<transformer_encoder_28/Encoder-1st-NormalizationLayer-2/beta@transformer_encoder_28/Encoder-SelfAttentionLayer-2/query/kernel>transformer_encoder_28/Encoder-SelfAttentionLayer-2/query/bias>transformer_encoder_28/Encoder-SelfAttentionLayer-2/key/kernel<transformer_encoder_28/Encoder-SelfAttentionLayer-2/key/bias@transformer_encoder_28/Encoder-SelfAttentionLayer-2/value/kernel>transformer_encoder_28/Encoder-SelfAttentionLayer-2/value/biasKtransformer_encoder_28/Encoder-SelfAttentionLayer-2/attention_output/kernelItransformer_encoder_28/Encoder-SelfAttentionLayer-2/attention_output/bias=transformer_encoder_28/Encoder-2nd-NormalizationLayer-2/gamma<transformer_encoder_28/Encoder-2nd-NormalizationLayer-2/beta:transformer_encoder_28/Encoder-FeedForwardLayer_1_2/kernel8transformer_encoder_28/Encoder-FeedForwardLayer_1_2/bias:transformer_encoder_28/Encoder-FeedForwardLayer_2_2/kernel8transformer_encoder_28/Encoder-FeedForwardLayer_2_2/bias=transformer_encoder_29/Encoder-1st-NormalizationLayer-3/gamma<transformer_encoder_29/Encoder-1st-NormalizationLayer-3/beta@transformer_encoder_29/Encoder-SelfAttentionLayer-3/query/kernel>transformer_encoder_29/Encoder-SelfAttentionLayer-3/query/bias>transformer_encoder_29/Encoder-SelfAttentionLayer-3/key/kernel<transformer_encoder_29/Encoder-SelfAttentionLayer-3/key/bias@transformer_encoder_29/Encoder-SelfAttentionLayer-3/value/kernel>transformer_encoder_29/Encoder-SelfAttentionLayer-3/value/biasKtransformer_encoder_29/Encoder-SelfAttentionLayer-3/attention_output/kernelItransformer_encoder_29/Encoder-SelfAttentionLayer-3/attention_output/bias=transformer_encoder_29/Encoder-2nd-NormalizationLayer-3/gamma<transformer_encoder_29/Encoder-2nd-NormalizationLayer-3/beta:transformer_encoder_29/Encoder-FeedForwardLayer_1_3/kernel8transformer_encoder_29/Encoder-FeedForwardLayer_1_3/bias:transformer_encoder_29/Encoder-FeedForwardLayer_2_3/kernel8transformer_encoder_29/Encoder-FeedForwardLayer_2_3/biasFinalLayerNorm/gammaFinalLayerNorm/beta#FullyConnectedLayerAreaRatio/kernel!FullyConnectedLayerAreaRatio/biasPredictionAreaRatio/kernelPredictionAreaRatio/biasPredictionStacks/kernelPredictionStacks/bias*E
Tin>
<2:*
Tout
2*
_collective_manager_ids
 *
_output_shapes

::*Z
_read_only_resource_inputs<
:8	
 !"#$%&'()*+,-./0123456789*0
config_proto 

CPU

GPU2*0J 8� *.
f)R'
%__inference_signature_wrapper_2630388

NoOpNoOp
��
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*ν
valueýB�� B��
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
layer-13
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures*
* 
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses* 
�
	variables
trainable_variables
regularization_losses
 	keras_api
!__call__
*"&call_and_return_all_conditional_losses
#self_attention_layer
$add1
%add2
&
layernorm1
'
layernorm2
(feed_forward_layer_1
)feed_forward_layer_2
*dropout_layer*
�
+	variables
,trainable_variables
-regularization_losses
.	keras_api
/__call__
*0&call_and_return_all_conditional_losses
1self_attention_layer
2add1
3add2
4
layernorm1
5
layernorm2
6feed_forward_layer_1
7feed_forward_layer_2
8dropout_layer*
�
9	variables
:trainable_variables
;regularization_losses
<	keras_api
=__call__
*>&call_and_return_all_conditional_losses
?self_attention_layer
@add1
Aadd2
B
layernorm1
C
layernorm2
Dfeed_forward_layer_1
Efeed_forward_layer_2
Fdropout_layer*
�
G	variables
Htrainable_variables
Iregularization_losses
J	keras_api
K__call__
*L&call_and_return_all_conditional_losses
Maxis
	Ngamma
Obeta*
* 
�
P	variables
Qtrainable_variables
Rregularization_losses
S	keras_api
T__call__
*U&call_and_return_all_conditional_losses* 
�
V	variables
Wtrainable_variables
Xregularization_losses
Y	keras_api
Z__call__
*[&call_and_return_all_conditional_losses* 
�
\	variables
]trainable_variables
^regularization_losses
_	keras_api
`__call__
*a&call_and_return_all_conditional_losses* 
�
b	variables
ctrainable_variables
dregularization_losses
e	keras_api
f__call__
*g&call_and_return_all_conditional_losses

hkernel
ibias*
�
j	variables
ktrainable_variables
lregularization_losses
m	keras_api
n__call__
*o&call_and_return_all_conditional_losses

pkernel
qbias*
�
r	variables
strainable_variables
tregularization_losses
u	keras_api
v__call__
*w&call_and_return_all_conditional_losses

xkernel
ybias*
�
z	variables
{trainable_variables
|regularization_losses
}	keras_api
~__call__
*&call_and_return_all_conditional_losses* 
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
N48
O49
h50
i51
p52
q53
x54
y55*
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
N48
O49
h50
i51
p52
q53
x54
y55*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

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
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses* 
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
	variables
trainable_variables
regularization_losses
!__call__
*"&call_and_return_all_conditional_losses
&""call_and_return_conditional_losses*

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
+	variables
,trainable_variables
-regularization_losses
/__call__
*0&call_and_return_all_conditional_losses
&0"call_and_return_conditional_losses*

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
9	variables
:trainable_variables
;regularization_losses
=__call__
*>&call_and_return_all_conditional_losses
&>"call_and_return_conditional_losses*

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
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�_random_generator* 

N0
O1*

N0
O1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
G	variables
Htrainable_variables
Iregularization_losses
K__call__
*L&call_and_return_all_conditional_losses
&L"call_and_return_conditional_losses*
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
P	variables
Qtrainable_variables
Rregularization_losses
T__call__
*U&call_and_return_all_conditional_losses
&U"call_and_return_conditional_losses* 

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
V	variables
Wtrainable_variables
Xregularization_losses
Z__call__
*[&call_and_return_all_conditional_losses
&["call_and_return_conditional_losses* 

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
\	variables
]trainable_variables
^regularization_losses
`__call__
*a&call_and_return_all_conditional_losses
&a"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

h0
i1*

h0
i1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
b	variables
ctrainable_variables
dregularization_losses
f__call__
*g&call_and_return_all_conditional_losses
&g"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
sm
VARIABLE_VALUE#FullyConnectedLayerAreaRatio/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
oi
VARIABLE_VALUE!FullyConnectedLayerAreaRatio/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*

p0
q1*

p0
q1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
j	variables
ktrainable_variables
lregularization_losses
n__call__
*o&call_and_return_all_conditional_losses
&o"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
ga
VARIABLE_VALUEPredictionStacks/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEPredictionStacks/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE*

x0
y1*

x0
y1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
r	variables
strainable_variables
tregularization_losses
v__call__
*w&call_and_return_all_conditional_losses
&w"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
jd
VARIABLE_VALUEPredictionAreaRatio/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUEPredictionAreaRatio/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
z	variables
{trainable_variables
|regularization_losses
~__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses* 
:
�trace_0
�trace_1
�trace_2
�trace_3* 
:
�trace_0
�trace_1
�trace_2
�trace_3* 
�z
VARIABLE_VALUE@transformer_encoder_27/Encoder-SelfAttentionLayer-1/query/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE>transformer_encoder_27/Encoder-SelfAttentionLayer-1/query/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE>transformer_encoder_27/Encoder-SelfAttentionLayer-1/key/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUE<transformer_encoder_27/Encoder-SelfAttentionLayer-1/key/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
�z
VARIABLE_VALUE@transformer_encoder_27/Encoder-SelfAttentionLayer-1/value/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE>transformer_encoder_27/Encoder-SelfAttentionLayer-1/value/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEKtransformer_encoder_27/Encoder-SelfAttentionLayer-1/attention_output/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEItransformer_encoder_27/Encoder-SelfAttentionLayer-1/attention_output/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUE=transformer_encoder_27/Encoder-1st-NormalizationLayer-1/gamma&variables/8/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUE<transformer_encoder_27/Encoder-1st-NormalizationLayer-1/beta&variables/9/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE=transformer_encoder_27/Encoder-2nd-NormalizationLayer-1/gamma'variables/10/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUE<transformer_encoder_27/Encoder-2nd-NormalizationLayer-1/beta'variables/11/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUE:transformer_encoder_27/Encoder-FeedForwardLayer_1_1/kernel'variables/12/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUE8transformer_encoder_27/Encoder-FeedForwardLayer_1_1/bias'variables/13/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUE:transformer_encoder_27/Encoder-FeedForwardLayer_2_1/kernel'variables/14/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUE8transformer_encoder_27/Encoder-FeedForwardLayer_2_1/bias'variables/15/.ATTRIBUTES/VARIABLE_VALUE*
�{
VARIABLE_VALUE@transformer_encoder_28/Encoder-SelfAttentionLayer-2/query/kernel'variables/16/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE>transformer_encoder_28/Encoder-SelfAttentionLayer-2/query/bias'variables/17/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE>transformer_encoder_28/Encoder-SelfAttentionLayer-2/key/kernel'variables/18/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUE<transformer_encoder_28/Encoder-SelfAttentionLayer-2/key/bias'variables/19/.ATTRIBUTES/VARIABLE_VALUE*
�{
VARIABLE_VALUE@transformer_encoder_28/Encoder-SelfAttentionLayer-2/value/kernel'variables/20/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE>transformer_encoder_28/Encoder-SelfAttentionLayer-2/value/bias'variables/21/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEKtransformer_encoder_28/Encoder-SelfAttentionLayer-2/attention_output/kernel'variables/22/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEItransformer_encoder_28/Encoder-SelfAttentionLayer-2/attention_output/bias'variables/23/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE=transformer_encoder_28/Encoder-1st-NormalizationLayer-2/gamma'variables/24/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUE<transformer_encoder_28/Encoder-1st-NormalizationLayer-2/beta'variables/25/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE=transformer_encoder_28/Encoder-2nd-NormalizationLayer-2/gamma'variables/26/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUE<transformer_encoder_28/Encoder-2nd-NormalizationLayer-2/beta'variables/27/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUE:transformer_encoder_28/Encoder-FeedForwardLayer_1_2/kernel'variables/28/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUE8transformer_encoder_28/Encoder-FeedForwardLayer_1_2/bias'variables/29/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUE:transformer_encoder_28/Encoder-FeedForwardLayer_2_2/kernel'variables/30/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUE8transformer_encoder_28/Encoder-FeedForwardLayer_2_2/bias'variables/31/.ATTRIBUTES/VARIABLE_VALUE*
�{
VARIABLE_VALUE@transformer_encoder_29/Encoder-SelfAttentionLayer-3/query/kernel'variables/32/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE>transformer_encoder_29/Encoder-SelfAttentionLayer-3/query/bias'variables/33/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE>transformer_encoder_29/Encoder-SelfAttentionLayer-3/key/kernel'variables/34/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUE<transformer_encoder_29/Encoder-SelfAttentionLayer-3/key/bias'variables/35/.ATTRIBUTES/VARIABLE_VALUE*
�{
VARIABLE_VALUE@transformer_encoder_29/Encoder-SelfAttentionLayer-3/value/kernel'variables/36/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE>transformer_encoder_29/Encoder-SelfAttentionLayer-3/value/bias'variables/37/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEKtransformer_encoder_29/Encoder-SelfAttentionLayer-3/attention_output/kernel'variables/38/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEItransformer_encoder_29/Encoder-SelfAttentionLayer-3/attention_output/bias'variables/39/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE=transformer_encoder_29/Encoder-1st-NormalizationLayer-3/gamma'variables/40/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUE<transformer_encoder_29/Encoder-1st-NormalizationLayer-3/beta'variables/41/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE=transformer_encoder_29/Encoder-2nd-NormalizationLayer-3/gamma'variables/42/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUE<transformer_encoder_29/Encoder-2nd-NormalizationLayer-3/beta'variables/43/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUE:transformer_encoder_29/Encoder-FeedForwardLayer_1_3/kernel'variables/44/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUE8transformer_encoder_29/Encoder-FeedForwardLayer_1_3/bias'variables/45/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUE:transformer_encoder_29/Encoder-FeedForwardLayer_2_3/kernel'variables/46/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUE8transformer_encoder_29/Encoder-FeedForwardLayer_2_3/bias'variables/47/.ATTRIBUTES/VARIABLE_VALUE*
* 
j
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
13*
* 
* 
* 
* 
* 
* 
* 
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
#0
$1
%2
&3
'4
(5
)6
*7*
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
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�partial_output_shape
�full_output_shape
�kernel
	�bias*
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
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 
* 
* 
* 
* 
<
10
21
32
43
54
65
76
87*
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
?0
@1
A2
B3
C4
D5
E6
F7*
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
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameFinalLayerNorm/gammaFinalLayerNorm/beta#FullyConnectedLayerAreaRatio/kernel!FullyConnectedLayerAreaRatio/biasPredictionStacks/kernelPredictionStacks/biasPredictionAreaRatio/kernelPredictionAreaRatio/bias@transformer_encoder_27/Encoder-SelfAttentionLayer-1/query/kernel>transformer_encoder_27/Encoder-SelfAttentionLayer-1/query/bias>transformer_encoder_27/Encoder-SelfAttentionLayer-1/key/kernel<transformer_encoder_27/Encoder-SelfAttentionLayer-1/key/bias@transformer_encoder_27/Encoder-SelfAttentionLayer-1/value/kernel>transformer_encoder_27/Encoder-SelfAttentionLayer-1/value/biasKtransformer_encoder_27/Encoder-SelfAttentionLayer-1/attention_output/kernelItransformer_encoder_27/Encoder-SelfAttentionLayer-1/attention_output/bias=transformer_encoder_27/Encoder-1st-NormalizationLayer-1/gamma<transformer_encoder_27/Encoder-1st-NormalizationLayer-1/beta=transformer_encoder_27/Encoder-2nd-NormalizationLayer-1/gamma<transformer_encoder_27/Encoder-2nd-NormalizationLayer-1/beta:transformer_encoder_27/Encoder-FeedForwardLayer_1_1/kernel8transformer_encoder_27/Encoder-FeedForwardLayer_1_1/bias:transformer_encoder_27/Encoder-FeedForwardLayer_2_1/kernel8transformer_encoder_27/Encoder-FeedForwardLayer_2_1/bias@transformer_encoder_28/Encoder-SelfAttentionLayer-2/query/kernel>transformer_encoder_28/Encoder-SelfAttentionLayer-2/query/bias>transformer_encoder_28/Encoder-SelfAttentionLayer-2/key/kernel<transformer_encoder_28/Encoder-SelfAttentionLayer-2/key/bias@transformer_encoder_28/Encoder-SelfAttentionLayer-2/value/kernel>transformer_encoder_28/Encoder-SelfAttentionLayer-2/value/biasKtransformer_encoder_28/Encoder-SelfAttentionLayer-2/attention_output/kernelItransformer_encoder_28/Encoder-SelfAttentionLayer-2/attention_output/bias=transformer_encoder_28/Encoder-1st-NormalizationLayer-2/gamma<transformer_encoder_28/Encoder-1st-NormalizationLayer-2/beta=transformer_encoder_28/Encoder-2nd-NormalizationLayer-2/gamma<transformer_encoder_28/Encoder-2nd-NormalizationLayer-2/beta:transformer_encoder_28/Encoder-FeedForwardLayer_1_2/kernel8transformer_encoder_28/Encoder-FeedForwardLayer_1_2/bias:transformer_encoder_28/Encoder-FeedForwardLayer_2_2/kernel8transformer_encoder_28/Encoder-FeedForwardLayer_2_2/bias@transformer_encoder_29/Encoder-SelfAttentionLayer-3/query/kernel>transformer_encoder_29/Encoder-SelfAttentionLayer-3/query/bias>transformer_encoder_29/Encoder-SelfAttentionLayer-3/key/kernel<transformer_encoder_29/Encoder-SelfAttentionLayer-3/key/bias@transformer_encoder_29/Encoder-SelfAttentionLayer-3/value/kernel>transformer_encoder_29/Encoder-SelfAttentionLayer-3/value/biasKtransformer_encoder_29/Encoder-SelfAttentionLayer-3/attention_output/kernelItransformer_encoder_29/Encoder-SelfAttentionLayer-3/attention_output/bias=transformer_encoder_29/Encoder-1st-NormalizationLayer-3/gamma<transformer_encoder_29/Encoder-1st-NormalizationLayer-3/beta=transformer_encoder_29/Encoder-2nd-NormalizationLayer-3/gamma<transformer_encoder_29/Encoder-2nd-NormalizationLayer-3/beta:transformer_encoder_29/Encoder-FeedForwardLayer_1_3/kernel8transformer_encoder_29/Encoder-FeedForwardLayer_1_3/bias:transformer_encoder_29/Encoder-FeedForwardLayer_2_3/kernel8transformer_encoder_29/Encoder-FeedForwardLayer_2_3/biasConst*E
Tin>
<2:*
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
 __inference__traced_save_2632327
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameFinalLayerNorm/gammaFinalLayerNorm/beta#FullyConnectedLayerAreaRatio/kernel!FullyConnectedLayerAreaRatio/biasPredictionStacks/kernelPredictionStacks/biasPredictionAreaRatio/kernelPredictionAreaRatio/bias@transformer_encoder_27/Encoder-SelfAttentionLayer-1/query/kernel>transformer_encoder_27/Encoder-SelfAttentionLayer-1/query/bias>transformer_encoder_27/Encoder-SelfAttentionLayer-1/key/kernel<transformer_encoder_27/Encoder-SelfAttentionLayer-1/key/bias@transformer_encoder_27/Encoder-SelfAttentionLayer-1/value/kernel>transformer_encoder_27/Encoder-SelfAttentionLayer-1/value/biasKtransformer_encoder_27/Encoder-SelfAttentionLayer-1/attention_output/kernelItransformer_encoder_27/Encoder-SelfAttentionLayer-1/attention_output/bias=transformer_encoder_27/Encoder-1st-NormalizationLayer-1/gamma<transformer_encoder_27/Encoder-1st-NormalizationLayer-1/beta=transformer_encoder_27/Encoder-2nd-NormalizationLayer-1/gamma<transformer_encoder_27/Encoder-2nd-NormalizationLayer-1/beta:transformer_encoder_27/Encoder-FeedForwardLayer_1_1/kernel8transformer_encoder_27/Encoder-FeedForwardLayer_1_1/bias:transformer_encoder_27/Encoder-FeedForwardLayer_2_1/kernel8transformer_encoder_27/Encoder-FeedForwardLayer_2_1/bias@transformer_encoder_28/Encoder-SelfAttentionLayer-2/query/kernel>transformer_encoder_28/Encoder-SelfAttentionLayer-2/query/bias>transformer_encoder_28/Encoder-SelfAttentionLayer-2/key/kernel<transformer_encoder_28/Encoder-SelfAttentionLayer-2/key/bias@transformer_encoder_28/Encoder-SelfAttentionLayer-2/value/kernel>transformer_encoder_28/Encoder-SelfAttentionLayer-2/value/biasKtransformer_encoder_28/Encoder-SelfAttentionLayer-2/attention_output/kernelItransformer_encoder_28/Encoder-SelfAttentionLayer-2/attention_output/bias=transformer_encoder_28/Encoder-1st-NormalizationLayer-2/gamma<transformer_encoder_28/Encoder-1st-NormalizationLayer-2/beta=transformer_encoder_28/Encoder-2nd-NormalizationLayer-2/gamma<transformer_encoder_28/Encoder-2nd-NormalizationLayer-2/beta:transformer_encoder_28/Encoder-FeedForwardLayer_1_2/kernel8transformer_encoder_28/Encoder-FeedForwardLayer_1_2/bias:transformer_encoder_28/Encoder-FeedForwardLayer_2_2/kernel8transformer_encoder_28/Encoder-FeedForwardLayer_2_2/bias@transformer_encoder_29/Encoder-SelfAttentionLayer-3/query/kernel>transformer_encoder_29/Encoder-SelfAttentionLayer-3/query/bias>transformer_encoder_29/Encoder-SelfAttentionLayer-3/key/kernel<transformer_encoder_29/Encoder-SelfAttentionLayer-3/key/bias@transformer_encoder_29/Encoder-SelfAttentionLayer-3/value/kernel>transformer_encoder_29/Encoder-SelfAttentionLayer-3/value/biasKtransformer_encoder_29/Encoder-SelfAttentionLayer-3/attention_output/kernelItransformer_encoder_29/Encoder-SelfAttentionLayer-3/attention_output/bias=transformer_encoder_29/Encoder-1st-NormalizationLayer-3/gamma<transformer_encoder_29/Encoder-1st-NormalizationLayer-3/beta=transformer_encoder_29/Encoder-2nd-NormalizationLayer-3/gamma<transformer_encoder_29/Encoder-2nd-NormalizationLayer-3/beta:transformer_encoder_29/Encoder-FeedForwardLayer_1_3/kernel8transformer_encoder_29/Encoder-FeedForwardLayer_1_3/bias:transformer_encoder_29/Encoder-FeedForwardLayer_2_3/kernel8transformer_encoder_29/Encoder-FeedForwardLayer_2_3/bias*D
Tin=
;29*
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
#__inference__traced_restore_2632504��/
�
R
6__inference_StandardizeTimeLimit_layer_call_fn_2631806

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
Q__inference_StandardizeTimeLimit_layer_call_and_return_conditional_losses_2628923`
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
_
C__inference_Output_layer_call_and_return_conditional_losses_2629018

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
_construction_contextkEagerRuntime**
_input_shapes
:���������P:S O
+
_output_shapes
:���������P
 
_user_specified_nameinputs
��
�
S__inference_transformer_encoder_28_layer_call_and_return_conditional_losses_2629409

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
dropout_28/IdentityIdentity-Encoder-FeedForwardLayer_2_2/BiasAdd:output:0*
T0*+
_output_shapes
:���������P	�
Encoder-2nd-AdditionLayer-2/addAddV2#Encoder-1st-AdditionLayer-2/add:z:0dropout_28/Identity:output:0*
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

�
P__inference_PredictionAreaRatio_layer_call_and_return_conditional_losses_2628966

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
Q__inference_StandardizeTimeLimit_layer_call_and_return_conditional_losses_2628923

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
�
�
M__inference_PredictionStacks_layer_call_and_return_conditional_losses_2629002

inputs3
!tensordot_readvariableop_resource:	-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Tensordot/ReadVariableOpz
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:	*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       S
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
::��Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:y
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:���������P	�
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������Pr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0|
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������PZ
SigmoidSigmoidBiasAdd:output:0*
T0*+
_output_shapes
:���������P^
IdentityIdentitySigmoid:y:0^NoOp*
T0*+
_output_shapes
:���������PV
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������P	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:($
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
M__inference_ConcatenateLayer_layer_call_and_return_conditional_losses_2628931

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
�
_
C__inference_Output_layer_call_and_return_conditional_losses_2631967

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
_construction_contextkEagerRuntime**
_input_shapes
:���������P:S O
+
_output_shapes
:���������P
 
_user_specified_nameinputs
��
�
S__inference_transformer_encoder_27_layer_call_and_return_conditional_losses_2630670

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
dropout_27/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?�
dropout_27/dropout/MulMul-Encoder-FeedForwardLayer_2_1/BiasAdd:output:0!dropout_27/dropout/Const:output:0*
T0*+
_output_shapes
:���������P	�
dropout_27/dropout/ShapeShape-Encoder-FeedForwardLayer_2_1/BiasAdd:output:0*
T0*
_output_shapes
::���
/dropout_27/dropout/random_uniform/RandomUniformRandomUniform!dropout_27/dropout/Shape:output:0*
T0*+
_output_shapes
:���������P	*
dtype0*
seed2*
seed��f
!dropout_27/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=�
dropout_27/dropout/GreaterEqualGreaterEqual8dropout_27/dropout/random_uniform/RandomUniform:output:0*dropout_27/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:���������P	_
dropout_27/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout_27/dropout/SelectV2SelectV2#dropout_27/dropout/GreaterEqual:z:0dropout_27/dropout/Mul:z:0#dropout_27/dropout/Const_1:output:0*
T0*+
_output_shapes
:���������P	�
Encoder-2nd-AdditionLayer-1/addAddV2#Encoder-1st-AdditionLayer-1/add:z:0$dropout_27/dropout/SelectV2:output:0*
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
�
y
]__inference_ReduceStackDimensionViaSummation_layer_call_and_return_conditional_losses_2629664

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
�
e
I__inference_MaskingLayer_layer_call_and_return_conditional_losses_2628190

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
C__inference_Output_layer_call_and_return_conditional_losses_2629696

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
S__inference_transformer_encoder_28_layer_call_and_return_conditional_losses_2628602

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
dropout_28/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?�
dropout_28/dropout/MulMul-Encoder-FeedForwardLayer_2_2/BiasAdd:output:0!dropout_28/dropout/Const:output:0*
T0*+
_output_shapes
:���������P	�
dropout_28/dropout/ShapeShape-Encoder-FeedForwardLayer_2_2/BiasAdd:output:0*
T0*
_output_shapes
::���
/dropout_28/dropout/random_uniform/RandomUniformRandomUniform!dropout_28/dropout/Shape:output:0*
T0*+
_output_shapes
:���������P	*
dtype0*
seed2*
seed��f
!dropout_28/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=�
dropout_28/dropout/GreaterEqualGreaterEqual8dropout_28/dropout/random_uniform/RandomUniform:output:0*dropout_28/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:���������P	_
dropout_28/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout_28/dropout/SelectV2SelectV2#dropout_28/dropout/GreaterEqual:z:0dropout_28/dropout/Mul:z:0#dropout_28/dropout/Const_1:output:0*
T0*+
_output_shapes
:���������P	�
Encoder-2nd-AdditionLayer-2/addAddV2#Encoder-1st-AdditionLayer-2/add:z:0$dropout_28/dropout/SelectV2:output:0*
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
^
B__inference_ReduceStackDimensionViaSummation_layer_call_fn_2631780

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
]__inference_ReduceStackDimensionViaSummation_layer_call_and_return_conditional_losses_2628913`
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
K__inference_FinalLayerNorm_layer_call_and_return_conditional_losses_2631775

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
�3
#__inference__traced_restore_2632504
file_prefix3
%assignvariableop_finallayernorm_gamma:	4
&assignvariableop_1_finallayernorm_beta:	H
6assignvariableop_2_fullyconnectedlayerarearatio_kernel:

B
4assignvariableop_3_fullyconnectedlayerarearatio_bias:
<
*assignvariableop_4_predictionstacks_kernel:	6
(assignvariableop_5_predictionstacks_bias:?
-assignvariableop_6_predictionarearatio_kernel:
9
+assignvariableop_7_predictionarearatio_bias:i
Sassignvariableop_8_transformer_encoder_27_encoder_selfattentionlayer_1_query_kernel:	c
Qassignvariableop_9_transformer_encoder_27_encoder_selfattentionlayer_1_query_bias:h
Rassignvariableop_10_transformer_encoder_27_encoder_selfattentionlayer_1_key_kernel:	b
Passignvariableop_11_transformer_encoder_27_encoder_selfattentionlayer_1_key_bias:j
Tassignvariableop_12_transformer_encoder_27_encoder_selfattentionlayer_1_value_kernel:	d
Rassignvariableop_13_transformer_encoder_27_encoder_selfattentionlayer_1_value_bias:u
_assignvariableop_14_transformer_encoder_27_encoder_selfattentionlayer_1_attention_output_kernel:	k
]assignvariableop_15_transformer_encoder_27_encoder_selfattentionlayer_1_attention_output_bias:	_
Qassignvariableop_16_transformer_encoder_27_encoder_1st_normalizationlayer_1_gamma:	^
Passignvariableop_17_transformer_encoder_27_encoder_1st_normalizationlayer_1_beta:	_
Qassignvariableop_18_transformer_encoder_27_encoder_2nd_normalizationlayer_1_gamma:	^
Passignvariableop_19_transformer_encoder_27_encoder_2nd_normalizationlayer_1_beta:	`
Nassignvariableop_20_transformer_encoder_27_encoder_feedforwardlayer_1_1_kernel:		Z
Lassignvariableop_21_transformer_encoder_27_encoder_feedforwardlayer_1_1_bias:	`
Nassignvariableop_22_transformer_encoder_27_encoder_feedforwardlayer_2_1_kernel:		Z
Lassignvariableop_23_transformer_encoder_27_encoder_feedforwardlayer_2_1_bias:	j
Tassignvariableop_24_transformer_encoder_28_encoder_selfattentionlayer_2_query_kernel:	d
Rassignvariableop_25_transformer_encoder_28_encoder_selfattentionlayer_2_query_bias:h
Rassignvariableop_26_transformer_encoder_28_encoder_selfattentionlayer_2_key_kernel:	b
Passignvariableop_27_transformer_encoder_28_encoder_selfattentionlayer_2_key_bias:j
Tassignvariableop_28_transformer_encoder_28_encoder_selfattentionlayer_2_value_kernel:	d
Rassignvariableop_29_transformer_encoder_28_encoder_selfattentionlayer_2_value_bias:u
_assignvariableop_30_transformer_encoder_28_encoder_selfattentionlayer_2_attention_output_kernel:	k
]assignvariableop_31_transformer_encoder_28_encoder_selfattentionlayer_2_attention_output_bias:	_
Qassignvariableop_32_transformer_encoder_28_encoder_1st_normalizationlayer_2_gamma:	^
Passignvariableop_33_transformer_encoder_28_encoder_1st_normalizationlayer_2_beta:	_
Qassignvariableop_34_transformer_encoder_28_encoder_2nd_normalizationlayer_2_gamma:	^
Passignvariableop_35_transformer_encoder_28_encoder_2nd_normalizationlayer_2_beta:	`
Nassignvariableop_36_transformer_encoder_28_encoder_feedforwardlayer_1_2_kernel:		Z
Lassignvariableop_37_transformer_encoder_28_encoder_feedforwardlayer_1_2_bias:	`
Nassignvariableop_38_transformer_encoder_28_encoder_feedforwardlayer_2_2_kernel:		Z
Lassignvariableop_39_transformer_encoder_28_encoder_feedforwardlayer_2_2_bias:	j
Tassignvariableop_40_transformer_encoder_29_encoder_selfattentionlayer_3_query_kernel:	d
Rassignvariableop_41_transformer_encoder_29_encoder_selfattentionlayer_3_query_bias:h
Rassignvariableop_42_transformer_encoder_29_encoder_selfattentionlayer_3_key_kernel:	b
Passignvariableop_43_transformer_encoder_29_encoder_selfattentionlayer_3_key_bias:j
Tassignvariableop_44_transformer_encoder_29_encoder_selfattentionlayer_3_value_kernel:	d
Rassignvariableop_45_transformer_encoder_29_encoder_selfattentionlayer_3_value_bias:u
_assignvariableop_46_transformer_encoder_29_encoder_selfattentionlayer_3_attention_output_kernel:	k
]assignvariableop_47_transformer_encoder_29_encoder_selfattentionlayer_3_attention_output_bias:	_
Qassignvariableop_48_transformer_encoder_29_encoder_1st_normalizationlayer_3_gamma:	^
Passignvariableop_49_transformer_encoder_29_encoder_1st_normalizationlayer_3_beta:	_
Qassignvariableop_50_transformer_encoder_29_encoder_2nd_normalizationlayer_3_gamma:	^
Passignvariableop_51_transformer_encoder_29_encoder_2nd_normalizationlayer_3_beta:	`
Nassignvariableop_52_transformer_encoder_29_encoder_feedforwardlayer_1_3_kernel:		Z
Lassignvariableop_53_transformer_encoder_29_encoder_feedforwardlayer_1_3_bias:	`
Nassignvariableop_54_transformer_encoder_29_encoder_feedforwardlayer_2_3_kernel:		Z
Lassignvariableop_55_transformer_encoder_29_encoder_feedforwardlayer_2_3_bias:	
identity_57��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_31�AssignVariableOp_32�AssignVariableOp_33�AssignVariableOp_34�AssignVariableOp_35�AssignVariableOp_36�AssignVariableOp_37�AssignVariableOp_38�AssignVariableOp_39�AssignVariableOp_4�AssignVariableOp_40�AssignVariableOp_41�AssignVariableOp_42�AssignVariableOp_43�AssignVariableOp_44�AssignVariableOp_45�AssignVariableOp_46�AssignVariableOp_47�AssignVariableOp_48�AssignVariableOp_49�AssignVariableOp_5�AssignVariableOp_50�AssignVariableOp_51�AssignVariableOp_52�AssignVariableOp_53�AssignVariableOp_54�AssignVariableOp_55�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:9*
dtype0*�
value�B�9B5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB'variables/26/.ATTRIBUTES/VARIABLE_VALUEB'variables/27/.ATTRIBUTES/VARIABLE_VALUEB'variables/28/.ATTRIBUTES/VARIABLE_VALUEB'variables/29/.ATTRIBUTES/VARIABLE_VALUEB'variables/30/.ATTRIBUTES/VARIABLE_VALUEB'variables/31/.ATTRIBUTES/VARIABLE_VALUEB'variables/32/.ATTRIBUTES/VARIABLE_VALUEB'variables/33/.ATTRIBUTES/VARIABLE_VALUEB'variables/34/.ATTRIBUTES/VARIABLE_VALUEB'variables/35/.ATTRIBUTES/VARIABLE_VALUEB'variables/36/.ATTRIBUTES/VARIABLE_VALUEB'variables/37/.ATTRIBUTES/VARIABLE_VALUEB'variables/38/.ATTRIBUTES/VARIABLE_VALUEB'variables/39/.ATTRIBUTES/VARIABLE_VALUEB'variables/40/.ATTRIBUTES/VARIABLE_VALUEB'variables/41/.ATTRIBUTES/VARIABLE_VALUEB'variables/42/.ATTRIBUTES/VARIABLE_VALUEB'variables/43/.ATTRIBUTES/VARIABLE_VALUEB'variables/44/.ATTRIBUTES/VARIABLE_VALUEB'variables/45/.ATTRIBUTES/VARIABLE_VALUEB'variables/46/.ATTRIBUTES/VARIABLE_VALUEB'variables/47/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:9*
dtype0*�
value|Bz9B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes�
�:::::::::::::::::::::::::::::::::::::::::::::::::::::::::*G
dtypes=
;29[
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
AssignVariableOp_4AssignVariableOp*assignvariableop_4_predictionstacks_kernelIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOp(assignvariableop_5_predictionstacks_biasIdentity_5:output:0"/device:CPU:0*&
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
AssignVariableOp_8AssignVariableOpSassignvariableop_8_transformer_encoder_27_encoder_selfattentionlayer_1_query_kernelIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOpQassignvariableop_9_transformer_encoder_27_encoder_selfattentionlayer_1_query_biasIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOpRassignvariableop_10_transformer_encoder_27_encoder_selfattentionlayer_1_key_kernelIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOpPassignvariableop_11_transformer_encoder_27_encoder_selfattentionlayer_1_key_biasIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOpTassignvariableop_12_transformer_encoder_27_encoder_selfattentionlayer_1_value_kernelIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOpRassignvariableop_13_transformer_encoder_27_encoder_selfattentionlayer_1_value_biasIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOp_assignvariableop_14_transformer_encoder_27_encoder_selfattentionlayer_1_attention_output_kernelIdentity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOp]assignvariableop_15_transformer_encoder_27_encoder_selfattentionlayer_1_attention_output_biasIdentity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOpQassignvariableop_16_transformer_encoder_27_encoder_1st_normalizationlayer_1_gammaIdentity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOpPassignvariableop_17_transformer_encoder_27_encoder_1st_normalizationlayer_1_betaIdentity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_18AssignVariableOpQassignvariableop_18_transformer_encoder_27_encoder_2nd_normalizationlayer_1_gammaIdentity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOpPassignvariableop_19_transformer_encoder_27_encoder_2nd_normalizationlayer_1_betaIdentity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_20AssignVariableOpNassignvariableop_20_transformer_encoder_27_encoder_feedforwardlayer_1_1_kernelIdentity_20:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_21AssignVariableOpLassignvariableop_21_transformer_encoder_27_encoder_feedforwardlayer_1_1_biasIdentity_21:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_22AssignVariableOpNassignvariableop_22_transformer_encoder_27_encoder_feedforwardlayer_2_1_kernelIdentity_22:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_23AssignVariableOpLassignvariableop_23_transformer_encoder_27_encoder_feedforwardlayer_2_1_biasIdentity_23:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_24AssignVariableOpTassignvariableop_24_transformer_encoder_28_encoder_selfattentionlayer_2_query_kernelIdentity_24:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_25AssignVariableOpRassignvariableop_25_transformer_encoder_28_encoder_selfattentionlayer_2_query_biasIdentity_25:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_26AssignVariableOpRassignvariableop_26_transformer_encoder_28_encoder_selfattentionlayer_2_key_kernelIdentity_26:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_27AssignVariableOpPassignvariableop_27_transformer_encoder_28_encoder_selfattentionlayer_2_key_biasIdentity_27:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_28AssignVariableOpTassignvariableop_28_transformer_encoder_28_encoder_selfattentionlayer_2_value_kernelIdentity_28:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_29AssignVariableOpRassignvariableop_29_transformer_encoder_28_encoder_selfattentionlayer_2_value_biasIdentity_29:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_30AssignVariableOp_assignvariableop_30_transformer_encoder_28_encoder_selfattentionlayer_2_attention_output_kernelIdentity_30:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_31AssignVariableOp]assignvariableop_31_transformer_encoder_28_encoder_selfattentionlayer_2_attention_output_biasIdentity_31:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_32AssignVariableOpQassignvariableop_32_transformer_encoder_28_encoder_1st_normalizationlayer_2_gammaIdentity_32:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_33AssignVariableOpPassignvariableop_33_transformer_encoder_28_encoder_1st_normalizationlayer_2_betaIdentity_33:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_34AssignVariableOpQassignvariableop_34_transformer_encoder_28_encoder_2nd_normalizationlayer_2_gammaIdentity_34:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_35AssignVariableOpPassignvariableop_35_transformer_encoder_28_encoder_2nd_normalizationlayer_2_betaIdentity_35:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_36AssignVariableOpNassignvariableop_36_transformer_encoder_28_encoder_feedforwardlayer_1_2_kernelIdentity_36:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_37AssignVariableOpLassignvariableop_37_transformer_encoder_28_encoder_feedforwardlayer_1_2_biasIdentity_37:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_38AssignVariableOpNassignvariableop_38_transformer_encoder_28_encoder_feedforwardlayer_2_2_kernelIdentity_38:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_39AssignVariableOpLassignvariableop_39_transformer_encoder_28_encoder_feedforwardlayer_2_2_biasIdentity_39:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_40AssignVariableOpTassignvariableop_40_transformer_encoder_29_encoder_selfattentionlayer_3_query_kernelIdentity_40:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_41AssignVariableOpRassignvariableop_41_transformer_encoder_29_encoder_selfattentionlayer_3_query_biasIdentity_41:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_42AssignVariableOpRassignvariableop_42_transformer_encoder_29_encoder_selfattentionlayer_3_key_kernelIdentity_42:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_43AssignVariableOpPassignvariableop_43_transformer_encoder_29_encoder_selfattentionlayer_3_key_biasIdentity_43:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_44AssignVariableOpTassignvariableop_44_transformer_encoder_29_encoder_selfattentionlayer_3_value_kernelIdentity_44:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_45AssignVariableOpRassignvariableop_45_transformer_encoder_29_encoder_selfattentionlayer_3_value_biasIdentity_45:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_46AssignVariableOp_assignvariableop_46_transformer_encoder_29_encoder_selfattentionlayer_3_attention_output_kernelIdentity_46:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_47AssignVariableOp]assignvariableop_47_transformer_encoder_29_encoder_selfattentionlayer_3_attention_output_biasIdentity_47:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_48AssignVariableOpQassignvariableop_48_transformer_encoder_29_encoder_1st_normalizationlayer_3_gammaIdentity_48:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_49AssignVariableOpPassignvariableop_49_transformer_encoder_29_encoder_1st_normalizationlayer_3_betaIdentity_49:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_50AssignVariableOpQassignvariableop_50_transformer_encoder_29_encoder_2nd_normalizationlayer_3_gammaIdentity_50:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_51AssignVariableOpPassignvariableop_51_transformer_encoder_29_encoder_2nd_normalizationlayer_3_betaIdentity_51:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_52AssignVariableOpNassignvariableop_52_transformer_encoder_29_encoder_feedforwardlayer_1_3_kernelIdentity_52:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_53AssignVariableOpLassignvariableop_53_transformer_encoder_29_encoder_feedforwardlayer_1_3_biasIdentity_53:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_54AssignVariableOpNassignvariableop_54_transformer_encoder_29_encoder_feedforwardlayer_2_3_kernelIdentity_54:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_55AssignVariableOpLassignvariableop_55_transformer_encoder_29_encoder_feedforwardlayer_2_3_biasIdentity_55:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 �

Identity_56Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_57IdentityIdentity_56:output:0^NoOp_1*
T0*
_output_shapes
: �	
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
_output_shapes
 "#
identity_57Identity_57:output:0*(
_construction_contextkEagerRuntime*�
_input_shapest
r: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2*
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
AssignVariableOp_55AssignVariableOp_552(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92$
AssignVariableOpAssignVariableOp:X8T
R
_user_specified_name:8transformer_encoder_29/Encoder-FeedForwardLayer_2_3/bias:Z7V
T
_user_specified_name<:transformer_encoder_29/Encoder-FeedForwardLayer_2_3/kernel:X6T
R
_user_specified_name:8transformer_encoder_29/Encoder-FeedForwardLayer_1_3/bias:Z5V
T
_user_specified_name<:transformer_encoder_29/Encoder-FeedForwardLayer_1_3/kernel:\4X
V
_user_specified_name><transformer_encoder_29/Encoder-2nd-NormalizationLayer-3/beta:]3Y
W
_user_specified_name?=transformer_encoder_29/Encoder-2nd-NormalizationLayer-3/gamma:\2X
V
_user_specified_name><transformer_encoder_29/Encoder-1st-NormalizationLayer-3/beta:]1Y
W
_user_specified_name?=transformer_encoder_29/Encoder-1st-NormalizationLayer-3/gamma:i0e
c
_user_specified_nameKItransformer_encoder_29/Encoder-SelfAttentionLayer-3/attention_output/bias:k/g
e
_user_specified_nameMKtransformer_encoder_29/Encoder-SelfAttentionLayer-3/attention_output/kernel:^.Z
X
_user_specified_name@>transformer_encoder_29/Encoder-SelfAttentionLayer-3/value/bias:`-\
Z
_user_specified_nameB@transformer_encoder_29/Encoder-SelfAttentionLayer-3/value/kernel:\,X
V
_user_specified_name><transformer_encoder_29/Encoder-SelfAttentionLayer-3/key/bias:^+Z
X
_user_specified_name@>transformer_encoder_29/Encoder-SelfAttentionLayer-3/key/kernel:^*Z
X
_user_specified_name@>transformer_encoder_29/Encoder-SelfAttentionLayer-3/query/bias:`)\
Z
_user_specified_nameB@transformer_encoder_29/Encoder-SelfAttentionLayer-3/query/kernel:X(T
R
_user_specified_name:8transformer_encoder_28/Encoder-FeedForwardLayer_2_2/bias:Z'V
T
_user_specified_name<:transformer_encoder_28/Encoder-FeedForwardLayer_2_2/kernel:X&T
R
_user_specified_name:8transformer_encoder_28/Encoder-FeedForwardLayer_1_2/bias:Z%V
T
_user_specified_name<:transformer_encoder_28/Encoder-FeedForwardLayer_1_2/kernel:\$X
V
_user_specified_name><transformer_encoder_28/Encoder-2nd-NormalizationLayer-2/beta:]#Y
W
_user_specified_name?=transformer_encoder_28/Encoder-2nd-NormalizationLayer-2/gamma:\"X
V
_user_specified_name><transformer_encoder_28/Encoder-1st-NormalizationLayer-2/beta:]!Y
W
_user_specified_name?=transformer_encoder_28/Encoder-1st-NormalizationLayer-2/gamma:i e
c
_user_specified_nameKItransformer_encoder_28/Encoder-SelfAttentionLayer-2/attention_output/bias:kg
e
_user_specified_nameMKtransformer_encoder_28/Encoder-SelfAttentionLayer-2/attention_output/kernel:^Z
X
_user_specified_name@>transformer_encoder_28/Encoder-SelfAttentionLayer-2/value/bias:`\
Z
_user_specified_nameB@transformer_encoder_28/Encoder-SelfAttentionLayer-2/value/kernel:\X
V
_user_specified_name><transformer_encoder_28/Encoder-SelfAttentionLayer-2/key/bias:^Z
X
_user_specified_name@>transformer_encoder_28/Encoder-SelfAttentionLayer-2/key/kernel:^Z
X
_user_specified_name@>transformer_encoder_28/Encoder-SelfAttentionLayer-2/query/bias:`\
Z
_user_specified_nameB@transformer_encoder_28/Encoder-SelfAttentionLayer-2/query/kernel:XT
R
_user_specified_name:8transformer_encoder_27/Encoder-FeedForwardLayer_2_1/bias:ZV
T
_user_specified_name<:transformer_encoder_27/Encoder-FeedForwardLayer_2_1/kernel:XT
R
_user_specified_name:8transformer_encoder_27/Encoder-FeedForwardLayer_1_1/bias:ZV
T
_user_specified_name<:transformer_encoder_27/Encoder-FeedForwardLayer_1_1/kernel:\X
V
_user_specified_name><transformer_encoder_27/Encoder-2nd-NormalizationLayer-1/beta:]Y
W
_user_specified_name?=transformer_encoder_27/Encoder-2nd-NormalizationLayer-1/gamma:\X
V
_user_specified_name><transformer_encoder_27/Encoder-1st-NormalizationLayer-1/beta:]Y
W
_user_specified_name?=transformer_encoder_27/Encoder-1st-NormalizationLayer-1/gamma:ie
c
_user_specified_nameKItransformer_encoder_27/Encoder-SelfAttentionLayer-1/attention_output/bias:kg
e
_user_specified_nameMKtransformer_encoder_27/Encoder-SelfAttentionLayer-1/attention_output/kernel:^Z
X
_user_specified_name@>transformer_encoder_27/Encoder-SelfAttentionLayer-1/value/bias:`\
Z
_user_specified_nameB@transformer_encoder_27/Encoder-SelfAttentionLayer-1/value/kernel:\X
V
_user_specified_name><transformer_encoder_27/Encoder-SelfAttentionLayer-1/key/bias:^Z
X
_user_specified_name@>transformer_encoder_27/Encoder-SelfAttentionLayer-1/key/kernel:^
Z
X
_user_specified_name@>transformer_encoder_27/Encoder-SelfAttentionLayer-1/query/bias:`	\
Z
_user_specified_nameB@transformer_encoder_27/Encoder-SelfAttentionLayer-1/query/kernel:84
2
_user_specified_namePredictionAreaRatio/bias::6
4
_user_specified_namePredictionAreaRatio/kernel:51
/
_user_specified_namePredictionStacks/bias:73
1
_user_specified_namePredictionStacks/kernel:A=
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
�
m
Q__inference_StandardizeTimeLimit_layer_call_and_return_conditional_losses_2631827

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
� 
�
K__inference_FinalLayerNorm_layer_call_and_return_conditional_losses_2628900

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
S__inference_transformer_encoder_29_layer_call_and_return_conditional_losses_2629617

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
dropout_29/IdentityIdentity-Encoder-FeedForwardLayer_2_3/BiasAdd:output:0*
T0*+
_output_shapes
:���������P	�
Encoder-2nd-AdditionLayer-3/addAddV2#Encoder-1st-AdditionLayer-3/add:z:0dropout_29/Identity:output:0*
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
Y__inference_FullyConnectedLayerAreaRatio_layer_call_and_return_conditional_losses_2631867

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
�.
�
%__inference_signature_wrapper_2630388
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

unknown_52:

unknown_53:	

unknown_54:
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
unknown_54*E
Tin>
<2:*
Tout
2*
_collective_manager_ids
 *
_output_shapes

::*Z
_read_only_resource_inputs<
:8	
 !"#$%&'()*+,-./0123456789*0
config_proto 

CPU

GPU2*0J 8� *+
f&R$
"__inference__wrapped_model_2628176`
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
�:���������P	:���������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:'9#
!
_user_specified_name	2630382:'8#
!
_user_specified_name	2630380:'7#
!
_user_specified_name	2630378:'6#
!
_user_specified_name	2630376:'5#
!
_user_specified_name	2630374:'4#
!
_user_specified_name	2630372:'3#
!
_user_specified_name	2630370:'2#
!
_user_specified_name	2630368:'1#
!
_user_specified_name	2630366:'0#
!
_user_specified_name	2630364:'/#
!
_user_specified_name	2630362:'.#
!
_user_specified_name	2630360:'-#
!
_user_specified_name	2630358:',#
!
_user_specified_name	2630356:'+#
!
_user_specified_name	2630354:'*#
!
_user_specified_name	2630352:')#
!
_user_specified_name	2630350:'(#
!
_user_specified_name	2630348:''#
!
_user_specified_name	2630346:'&#
!
_user_specified_name	2630344:'%#
!
_user_specified_name	2630342:'$#
!
_user_specified_name	2630340:'##
!
_user_specified_name	2630338:'"#
!
_user_specified_name	2630336:'!#
!
_user_specified_name	2630334:' #
!
_user_specified_name	2630332:'#
!
_user_specified_name	2630330:'#
!
_user_specified_name	2630328:'#
!
_user_specified_name	2630326:'#
!
_user_specified_name	2630324:'#
!
_user_specified_name	2630322:'#
!
_user_specified_name	2630320:'#
!
_user_specified_name	2630318:'#
!
_user_specified_name	2630316:'#
!
_user_specified_name	2630314:'#
!
_user_specified_name	2630312:'#
!
_user_specified_name	2630310:'#
!
_user_specified_name	2630308:'#
!
_user_specified_name	2630306:'#
!
_user_specified_name	2630304:'#
!
_user_specified_name	2630302:'#
!
_user_specified_name	2630300:'#
!
_user_specified_name	2630298:'#
!
_user_specified_name	2630296:'#
!
_user_specified_name	2630294:'#
!
_user_specified_name	2630292:'#
!
_user_specified_name	2630290:'
#
!
_user_specified_name	2630288:'	#
!
_user_specified_name	2630286:'#
!
_user_specified_name	2630284:'#
!
_user_specified_name	2630282:'#
!
_user_specified_name	2630280:'#
!
_user_specified_name	2630278:'#
!
_user_specified_name	2630276:'#
!
_user_specified_name	2630274:'#
!
_user_specified_name	2630272:WS
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
�
_
C__inference_Output_layer_call_and_return_conditional_losses_2629012

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
S__inference_transformer_encoder_29_layer_call_and_return_conditional_losses_2631724

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
dropout_29/IdentityIdentity-Encoder-FeedForwardLayer_2_3/BiasAdd:output:0*
T0*+
_output_shapes
:���������P	�
Encoder-2nd-AdditionLayer-3/addAddV2#Encoder-1st-AdditionLayer-3/add:z:0dropout_29/Identity:output:0*
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
8__inference_transformer_encoder_29_layer_call_fn_2631323

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
S__inference_transformer_encoder_29_layer_call_and_return_conditional_losses_2628824s
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
_user_specified_name	2631317:'#
!
_user_specified_name	2631315:'#
!
_user_specified_name	2631313:'#
!
_user_specified_name	2631311:'#
!
_user_specified_name	2631309:'#
!
_user_specified_name	2631307:'
#
!
_user_specified_name	2631305:'	#
!
_user_specified_name	2631303:'#
!
_user_specified_name	2631301:'#
!
_user_specified_name	2631299:'#
!
_user_specified_name	2631297:'#
!
_user_specified_name	2631295:'#
!
_user_specified_name	2631293:'#
!
_user_specified_name	2631291:'#
!
_user_specified_name	2631289:'#
!
_user_specified_name	2631287:S O
+
_output_shapes
:���������P	
 
_user_specified_nameinputs
��
�
S__inference_transformer_encoder_27_layer_call_and_return_conditional_losses_2629201

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
dropout_27/IdentityIdentity-Encoder-FeedForwardLayer_2_1/BiasAdd:output:0*
T0*+
_output_shapes
:���������P	�
Encoder-2nd-AdditionLayer-1/addAddV2#Encoder-1st-AdditionLayer-1/add:z:0dropout_27/Identity:output:0*
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
�
S__inference_transformer_encoder_29_layer_call_and_return_conditional_losses_2631550

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
dropout_29/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?�
dropout_29/dropout/MulMul-Encoder-FeedForwardLayer_2_3/BiasAdd:output:0!dropout_29/dropout/Const:output:0*
T0*+
_output_shapes
:���������P	�
dropout_29/dropout/ShapeShape-Encoder-FeedForwardLayer_2_3/BiasAdd:output:0*
T0*
_output_shapes
::���
/dropout_29/dropout/random_uniform/RandomUniformRandomUniform!dropout_29/dropout/Shape:output:0*
T0*+
_output_shapes
:���������P	*
dtype0*
seed2*
seed��f
!dropout_29/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=�
dropout_29/dropout/GreaterEqualGreaterEqual8dropout_29/dropout/random_uniform/RandomUniform:output:0*dropout_29/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:���������P	_
dropout_29/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout_29/dropout/SelectV2SelectV2#dropout_29/dropout/GreaterEqual:z:0dropout_29/dropout/Mul:z:0#dropout_29/dropout/Const_1:output:0*
T0*+
_output_shapes
:���������P	�
Encoder-2nd-AdditionLayer-3/addAddV2#Encoder-1st-AdditionLayer-3/add:z:0$dropout_29/dropout/SelectV2:output:0*
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
�

�
P__inference_PredictionAreaRatio_layer_call_and_return_conditional_losses_2631927

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
�
_
C__inference_Output_layer_call_and_return_conditional_losses_2629702

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
_construction_contextkEagerRuntime**
_input_shapes
:���������P:S O
+
_output_shapes
:���������P
 
_user_specified_nameinputs
�
^
2__inference_ConcatenateLayer_layer_call_fn_2631833
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
M__inference_ConcatenateLayer_layer_call_and_return_conditional_losses_2628931`
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
�
�
8__inference_transformer_encoder_28_layer_call_fn_2630883

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
S__inference_transformer_encoder_28_layer_call_and_return_conditional_losses_2628602s
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
_user_specified_name	2630877:'#
!
_user_specified_name	2630875:'#
!
_user_specified_name	2630873:'#
!
_user_specified_name	2630871:'#
!
_user_specified_name	2630869:'#
!
_user_specified_name	2630867:'
#
!
_user_specified_name	2630865:'	#
!
_user_specified_name	2630863:'#
!
_user_specified_name	2630861:'#
!
_user_specified_name	2630859:'#
!
_user_specified_name	2630857:'#
!
_user_specified_name	2630855:'#
!
_user_specified_name	2630853:'#
!
_user_specified_name	2630851:'#
!
_user_specified_name	2630849:'#
!
_user_specified_name	2630847:S O
+
_output_shapes
:���������P	
 
_user_specified_nameinputs
�
y
]__inference_ReduceStackDimensionViaSummation_layer_call_and_return_conditional_losses_2628913

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
�
�
>__inference_FullyConnectedLayerAreaRatio_layer_call_fn_2631849

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
Y__inference_FullyConnectedLayerAreaRatio_layer_call_and_return_conditional_losses_2628950o
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
_user_specified_name	2631845:'#
!
_user_specified_name	2631843:O K
'
_output_shapes
:���������

 
_user_specified_nameinputs
��
�
S__inference_transformer_encoder_28_layer_call_and_return_conditional_losses_2631284

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
dropout_28/IdentityIdentity-Encoder-FeedForwardLayer_2_2/BiasAdd:output:0*
T0*+
_output_shapes
:���������P	�
Encoder-2nd-AdditionLayer-2/addAddV2#Encoder-1st-AdditionLayer-2/add:z:0dropout_28/Identity:output:0*
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
(__inference_Output_layer_call_fn_2631932

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
C__inference_Output_layer_call_and_return_conditional_losses_2629012Q
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
�
�
M__inference_PredictionStacks_layer_call_and_return_conditional_losses_2631907

inputs3
!tensordot_readvariableop_resource:	-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Tensordot/ReadVariableOpz
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:	*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       S
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
::��Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:y
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:���������P	�
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������Pr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0|
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������PZ
SigmoidSigmoidBiasAdd:output:0*
T0*+
_output_shapes
:���������P^
IdentityIdentitySigmoid:y:0^NoOp*
T0*+
_output_shapes
:���������PV
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������P	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:($
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
y
M__inference_ConcatenateLayer_layer_call_and_return_conditional_losses_2631840
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
8__inference_transformer_encoder_27_layer_call_fn_2630443

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
S__inference_transformer_encoder_27_layer_call_and_return_conditional_losses_2628380s
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
_user_specified_name	2630437:'#
!
_user_specified_name	2630435:'#
!
_user_specified_name	2630433:'#
!
_user_specified_name	2630431:'#
!
_user_specified_name	2630429:'#
!
_user_specified_name	2630427:'
#
!
_user_specified_name	2630425:'	#
!
_user_specified_name	2630423:'#
!
_user_specified_name	2630421:'#
!
_user_specified_name	2630419:'#
!
_user_specified_name	2630417:'#
!
_user_specified_name	2630415:'#
!
_user_specified_name	2630413:'#
!
_user_specified_name	2630411:'#
!
_user_specified_name	2630409:'#
!
_user_specified_name	2630407:S O
+
_output_shapes
:���������P	
 
_user_specified_nameinputs
�
�
0__inference_FinalLayerNorm_layer_call_fn_2631733

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
K__inference_FinalLayerNorm_layer_call_and_return_conditional_losses_2628900s
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
_user_specified_name	2631729:'#
!
_user_specified_name	2631727:S O
+
_output_shapes
:���������P	
 
_user_specified_nameinputs
�
D
(__inference_Output_layer_call_fn_2631942

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
C__inference_Output_layer_call_and_return_conditional_losses_2629018Q
IdentityIdentityPartitionedCall:output:0*
T0*
_output_shapes
:"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������P:S O
+
_output_shapes
:���������P
 
_user_specified_nameinputs
�
_
C__inference_Output_layer_call_and_return_conditional_losses_2631952

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
(__inference_Output_layer_call_fn_2631937

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
C__inference_Output_layer_call_and_return_conditional_losses_2629696Q
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
�
�
8__inference_transformer_encoder_28_layer_call_fn_2630922

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
S__inference_transformer_encoder_28_layer_call_and_return_conditional_losses_2629409s
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
_user_specified_name	2630916:'#
!
_user_specified_name	2630914:'#
!
_user_specified_name	2630912:'#
!
_user_specified_name	2630910:'#
!
_user_specified_name	2630908:'#
!
_user_specified_name	2630906:'
#
!
_user_specified_name	2630904:'	#
!
_user_specified_name	2630902:'#
!
_user_specified_name	2630900:'#
!
_user_specified_name	2630898:'#
!
_user_specified_name	2630896:'#
!
_user_specified_name	2630894:'#
!
_user_specified_name	2630892:'#
!
_user_specified_name	2630890:'#
!
_user_specified_name	2630888:'#
!
_user_specified_name	2630886:S O
+
_output_shapes
:���������P	
 
_user_specified_nameinputs
�.
�
)__inference_model_9_layer_call_fn_2629946
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

unknown_52:

unknown_53:	

unknown_54:
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
unknown_54*E
Tin>
<2:*
Tout
2*
_collective_manager_ids
 *
_output_shapes

::*Z
_read_only_resource_inputs<
:8	
 !"#$%&'()*+,-./0123456789*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_model_9_layer_call_and_return_conditional_losses_2629706`
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
�:���������P	:���������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:'9#
!
_user_specified_name	2629940:'8#
!
_user_specified_name	2629938:'7#
!
_user_specified_name	2629936:'6#
!
_user_specified_name	2629934:'5#
!
_user_specified_name	2629932:'4#
!
_user_specified_name	2629930:'3#
!
_user_specified_name	2629928:'2#
!
_user_specified_name	2629926:'1#
!
_user_specified_name	2629924:'0#
!
_user_specified_name	2629922:'/#
!
_user_specified_name	2629920:'.#
!
_user_specified_name	2629918:'-#
!
_user_specified_name	2629916:',#
!
_user_specified_name	2629914:'+#
!
_user_specified_name	2629912:'*#
!
_user_specified_name	2629910:')#
!
_user_specified_name	2629908:'(#
!
_user_specified_name	2629906:''#
!
_user_specified_name	2629904:'&#
!
_user_specified_name	2629902:'%#
!
_user_specified_name	2629900:'$#
!
_user_specified_name	2629898:'##
!
_user_specified_name	2629896:'"#
!
_user_specified_name	2629894:'!#
!
_user_specified_name	2629892:' #
!
_user_specified_name	2629890:'#
!
_user_specified_name	2629888:'#
!
_user_specified_name	2629886:'#
!
_user_specified_name	2629884:'#
!
_user_specified_name	2629882:'#
!
_user_specified_name	2629880:'#
!
_user_specified_name	2629878:'#
!
_user_specified_name	2629876:'#
!
_user_specified_name	2629874:'#
!
_user_specified_name	2629872:'#
!
_user_specified_name	2629870:'#
!
_user_specified_name	2629868:'#
!
_user_specified_name	2629866:'#
!
_user_specified_name	2629864:'#
!
_user_specified_name	2629862:'#
!
_user_specified_name	2629860:'#
!
_user_specified_name	2629858:'#
!
_user_specified_name	2629856:'#
!
_user_specified_name	2629854:'#
!
_user_specified_name	2629852:'#
!
_user_specified_name	2629850:'#
!
_user_specified_name	2629848:'
#
!
_user_specified_name	2629846:'	#
!
_user_specified_name	2629844:'#
!
_user_specified_name	2629842:'#
!
_user_specified_name	2629840:'#
!
_user_specified_name	2629838:'#
!
_user_specified_name	2629836:'#
!
_user_specified_name	2629834:'#
!
_user_specified_name	2629832:'#
!
_user_specified_name	2629830:WS
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
2__inference_PredictionStacks_layer_call_fn_2631876

inputs
unknown:	
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������P*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *V
fQRO
M__inference_PredictionStacks_layer_call_and_return_conditional_losses_2629002s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������P<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������P	: : 22
StatefulPartitionedCallStatefulPartitionedCall:'#
!
_user_specified_name	2631872:'#
!
_user_specified_name	2631870:S O
+
_output_shapes
:���������P	
 
_user_specified_nameinputs
�.
�
)__inference_model_9_layer_call_fn_2629826
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

unknown_52:

unknown_53:	

unknown_54:
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
unknown_54*E
Tin>
<2:*
Tout
2*
_collective_manager_ids
 *
_output_shapes

::*Z
_read_only_resource_inputs<
:8	
 !"#$%&'()*+,-./0123456789*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_model_9_layer_call_and_return_conditional_losses_2629022`
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
�:���������P	:���������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:'9#
!
_user_specified_name	2629820:'8#
!
_user_specified_name	2629818:'7#
!
_user_specified_name	2629816:'6#
!
_user_specified_name	2629814:'5#
!
_user_specified_name	2629812:'4#
!
_user_specified_name	2629810:'3#
!
_user_specified_name	2629808:'2#
!
_user_specified_name	2629806:'1#
!
_user_specified_name	2629804:'0#
!
_user_specified_name	2629802:'/#
!
_user_specified_name	2629800:'.#
!
_user_specified_name	2629798:'-#
!
_user_specified_name	2629796:',#
!
_user_specified_name	2629794:'+#
!
_user_specified_name	2629792:'*#
!
_user_specified_name	2629790:')#
!
_user_specified_name	2629788:'(#
!
_user_specified_name	2629786:''#
!
_user_specified_name	2629784:'&#
!
_user_specified_name	2629782:'%#
!
_user_specified_name	2629780:'$#
!
_user_specified_name	2629778:'##
!
_user_specified_name	2629776:'"#
!
_user_specified_name	2629774:'!#
!
_user_specified_name	2629772:' #
!
_user_specified_name	2629770:'#
!
_user_specified_name	2629768:'#
!
_user_specified_name	2629766:'#
!
_user_specified_name	2629764:'#
!
_user_specified_name	2629762:'#
!
_user_specified_name	2629760:'#
!
_user_specified_name	2629758:'#
!
_user_specified_name	2629756:'#
!
_user_specified_name	2629754:'#
!
_user_specified_name	2629752:'#
!
_user_specified_name	2629750:'#
!
_user_specified_name	2629748:'#
!
_user_specified_name	2629746:'#
!
_user_specified_name	2629744:'#
!
_user_specified_name	2629742:'#
!
_user_specified_name	2629740:'#
!
_user_specified_name	2629738:'#
!
_user_specified_name	2629736:'#
!
_user_specified_name	2629734:'#
!
_user_specified_name	2629732:'#
!
_user_specified_name	2629730:'#
!
_user_specified_name	2629728:'
#
!
_user_specified_name	2629726:'	#
!
_user_specified_name	2629724:'#
!
_user_specified_name	2629722:'#
!
_user_specified_name	2629720:'#
!
_user_specified_name	2629718:'#
!
_user_specified_name	2629716:'#
!
_user_specified_name	2629714:'#
!
_user_specified_name	2629712:'#
!
_user_specified_name	2629710:WS
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
e
I__inference_MaskingLayer_layer_call_and_return_conditional_losses_2630404

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
�q
�
D__inference_model_9_layer_call_and_return_conditional_losses_2629706
stacklevelinputfeatures
timelimitinput,
transformer_encoder_27_2629202:	,
transformer_encoder_27_2629204:	4
transformer_encoder_27_2629206:	0
transformer_encoder_27_2629208:4
transformer_encoder_27_2629210:	0
transformer_encoder_27_2629212:4
transformer_encoder_27_2629214:	0
transformer_encoder_27_2629216:4
transformer_encoder_27_2629218:	,
transformer_encoder_27_2629220:	,
transformer_encoder_27_2629222:	,
transformer_encoder_27_2629224:	0
transformer_encoder_27_2629226:		,
transformer_encoder_27_2629228:	0
transformer_encoder_27_2629230:		,
transformer_encoder_27_2629232:	,
transformer_encoder_28_2629410:	,
transformer_encoder_28_2629412:	4
transformer_encoder_28_2629414:	0
transformer_encoder_28_2629416:4
transformer_encoder_28_2629418:	0
transformer_encoder_28_2629420:4
transformer_encoder_28_2629422:	0
transformer_encoder_28_2629424:4
transformer_encoder_28_2629426:	,
transformer_encoder_28_2629428:	,
transformer_encoder_28_2629430:	,
transformer_encoder_28_2629432:	0
transformer_encoder_28_2629434:		,
transformer_encoder_28_2629436:	0
transformer_encoder_28_2629438:		,
transformer_encoder_28_2629440:	,
transformer_encoder_29_2629618:	,
transformer_encoder_29_2629620:	4
transformer_encoder_29_2629622:	0
transformer_encoder_29_2629624:4
transformer_encoder_29_2629626:	0
transformer_encoder_29_2629628:4
transformer_encoder_29_2629630:	0
transformer_encoder_29_2629632:4
transformer_encoder_29_2629634:	,
transformer_encoder_29_2629636:	,
transformer_encoder_29_2629638:	,
transformer_encoder_29_2629640:	0
transformer_encoder_29_2629642:		,
transformer_encoder_29_2629644:	0
transformer_encoder_29_2629646:		,
transformer_encoder_29_2629648:	$
finallayernorm_2629652:	$
finallayernorm_2629654:	6
$fullyconnectedlayerarearatio_2629677:

2
$fullyconnectedlayerarearatio_2629679:
-
predictionarearatio_2629682:
)
predictionarearatio_2629684:*
predictionstacks_2629687:	&
predictionstacks_2629689:
identity

identity_1��&FinalLayerNorm/StatefulPartitionedCall�4FullyConnectedLayerAreaRatio/StatefulPartitionedCall�+PredictionAreaRatio/StatefulPartitionedCall�(PredictionStacks/StatefulPartitionedCall�.transformer_encoder_27/StatefulPartitionedCall�.transformer_encoder_28/StatefulPartitionedCall�.transformer_encoder_29/StatefulPartitionedCall�
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
I__inference_MaskingLayer_layer_call_and_return_conditional_losses_2628190�
transformer_encoder_27/CastCast%MaskingLayer/PartitionedCall:output:0*

DstT0*

SrcT0*+
_output_shapes
:���������P	�
.transformer_encoder_27/StatefulPartitionedCallStatefulPartitionedCalltransformer_encoder_27/Cast:y:0transformer_encoder_27_2629202transformer_encoder_27_2629204transformer_encoder_27_2629206transformer_encoder_27_2629208transformer_encoder_27_2629210transformer_encoder_27_2629212transformer_encoder_27_2629214transformer_encoder_27_2629216transformer_encoder_27_2629218transformer_encoder_27_2629220transformer_encoder_27_2629222transformer_encoder_27_2629224transformer_encoder_27_2629226transformer_encoder_27_2629228transformer_encoder_27_2629230transformer_encoder_27_2629232*
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
S__inference_transformer_encoder_27_layer_call_and_return_conditional_losses_2629201�
.transformer_encoder_28/StatefulPartitionedCallStatefulPartitionedCall7transformer_encoder_27/StatefulPartitionedCall:output:0transformer_encoder_28_2629410transformer_encoder_28_2629412transformer_encoder_28_2629414transformer_encoder_28_2629416transformer_encoder_28_2629418transformer_encoder_28_2629420transformer_encoder_28_2629422transformer_encoder_28_2629424transformer_encoder_28_2629426transformer_encoder_28_2629428transformer_encoder_28_2629430transformer_encoder_28_2629432transformer_encoder_28_2629434transformer_encoder_28_2629436transformer_encoder_28_2629438transformer_encoder_28_2629440*
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
S__inference_transformer_encoder_28_layer_call_and_return_conditional_losses_2629409�
.transformer_encoder_29/StatefulPartitionedCallStatefulPartitionedCall7transformer_encoder_28/StatefulPartitionedCall:output:0transformer_encoder_29_2629618transformer_encoder_29_2629620transformer_encoder_29_2629622transformer_encoder_29_2629624transformer_encoder_29_2629626transformer_encoder_29_2629628transformer_encoder_29_2629630transformer_encoder_29_2629632transformer_encoder_29_2629634transformer_encoder_29_2629636transformer_encoder_29_2629638transformer_encoder_29_2629640transformer_encoder_29_2629642transformer_encoder_29_2629644transformer_encoder_29_2629646transformer_encoder_29_2629648*
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
S__inference_transformer_encoder_29_layer_call_and_return_conditional_losses_2629617�
&FinalLayerNorm/StatefulPartitionedCallStatefulPartitionedCall7transformer_encoder_29/StatefulPartitionedCall:output:0finallayernorm_2629652finallayernorm_2629654*
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
K__inference_FinalLayerNorm_layer_call_and_return_conditional_losses_2628900�
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
]__inference_ReduceStackDimensionViaSummation_layer_call_and_return_conditional_losses_2629664r
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
Q__inference_StandardizeTimeLimit_layer_call_and_return_conditional_losses_2629674�
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
M__inference_ConcatenateLayer_layer_call_and_return_conditional_losses_2628931�
4FullyConnectedLayerAreaRatio/StatefulPartitionedCallStatefulPartitionedCall)ConcatenateLayer/PartitionedCall:output:0$fullyconnectedlayerarearatio_2629677$fullyconnectedlayerarearatio_2629679*
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
Y__inference_FullyConnectedLayerAreaRatio_layer_call_and_return_conditional_losses_2628950�
+PredictionAreaRatio/StatefulPartitionedCallStatefulPartitionedCall=FullyConnectedLayerAreaRatio/StatefulPartitionedCall:output:0predictionarearatio_2629682predictionarearatio_2629684*
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
P__inference_PredictionAreaRatio_layer_call_and_return_conditional_losses_2628966�
(PredictionStacks/StatefulPartitionedCallStatefulPartitionedCall/FinalLayerNorm/StatefulPartitionedCall:output:0predictionstacks_2629687predictionstacks_2629689*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������P*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *V
fQRO
M__inference_PredictionStacks_layer_call_and_return_conditional_losses_2629002�
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
C__inference_Output_layer_call_and_return_conditional_losses_2629696�
Output/PartitionedCall_1PartitionedCall1PredictionStacks/StatefulPartitionedCall:output:0*
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
C__inference_Output_layer_call_and_return_conditional_losses_2629702a
IdentityIdentity!Output/PartitionedCall_1:output:0^NoOp*
T0*
_output_shapes
:a

Identity_1IdentityOutput/PartitionedCall:output:0^NoOp*
T0*
_output_shapes
:�
NoOpNoOp'^FinalLayerNorm/StatefulPartitionedCall5^FullyConnectedLayerAreaRatio/StatefulPartitionedCall,^PredictionAreaRatio/StatefulPartitionedCall)^PredictionStacks/StatefulPartitionedCall/^transformer_encoder_27/StatefulPartitionedCall/^transformer_encoder_28/StatefulPartitionedCall/^transformer_encoder_29/StatefulPartitionedCall*
_output_shapes
 "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:���������P	:���������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2P
&FinalLayerNorm/StatefulPartitionedCall&FinalLayerNorm/StatefulPartitionedCall2l
4FullyConnectedLayerAreaRatio/StatefulPartitionedCall4FullyConnectedLayerAreaRatio/StatefulPartitionedCall2Z
+PredictionAreaRatio/StatefulPartitionedCall+PredictionAreaRatio/StatefulPartitionedCall2T
(PredictionStacks/StatefulPartitionedCall(PredictionStacks/StatefulPartitionedCall2`
.transformer_encoder_27/StatefulPartitionedCall.transformer_encoder_27/StatefulPartitionedCall2`
.transformer_encoder_28/StatefulPartitionedCall.transformer_encoder_28/StatefulPartitionedCall2`
.transformer_encoder_29/StatefulPartitionedCall.transformer_encoder_29/StatefulPartitionedCall:'9#
!
_user_specified_name	2629689:'8#
!
_user_specified_name	2629687:'7#
!
_user_specified_name	2629684:'6#
!
_user_specified_name	2629682:'5#
!
_user_specified_name	2629679:'4#
!
_user_specified_name	2629677:'3#
!
_user_specified_name	2629654:'2#
!
_user_specified_name	2629652:'1#
!
_user_specified_name	2629648:'0#
!
_user_specified_name	2629646:'/#
!
_user_specified_name	2629644:'.#
!
_user_specified_name	2629642:'-#
!
_user_specified_name	2629640:',#
!
_user_specified_name	2629638:'+#
!
_user_specified_name	2629636:'*#
!
_user_specified_name	2629634:')#
!
_user_specified_name	2629632:'(#
!
_user_specified_name	2629630:''#
!
_user_specified_name	2629628:'&#
!
_user_specified_name	2629626:'%#
!
_user_specified_name	2629624:'$#
!
_user_specified_name	2629622:'##
!
_user_specified_name	2629620:'"#
!
_user_specified_name	2629618:'!#
!
_user_specified_name	2629440:' #
!
_user_specified_name	2629438:'#
!
_user_specified_name	2629436:'#
!
_user_specified_name	2629434:'#
!
_user_specified_name	2629432:'#
!
_user_specified_name	2629430:'#
!
_user_specified_name	2629428:'#
!
_user_specified_name	2629426:'#
!
_user_specified_name	2629424:'#
!
_user_specified_name	2629422:'#
!
_user_specified_name	2629420:'#
!
_user_specified_name	2629418:'#
!
_user_specified_name	2629416:'#
!
_user_specified_name	2629414:'#
!
_user_specified_name	2629412:'#
!
_user_specified_name	2629410:'#
!
_user_specified_name	2629232:'#
!
_user_specified_name	2629230:'#
!
_user_specified_name	2629228:'#
!
_user_specified_name	2629226:'#
!
_user_specified_name	2629224:'#
!
_user_specified_name	2629222:'#
!
_user_specified_name	2629220:'
#
!
_user_specified_name	2629218:'	#
!
_user_specified_name	2629216:'#
!
_user_specified_name	2629214:'#
!
_user_specified_name	2629212:'#
!
_user_specified_name	2629210:'#
!
_user_specified_name	2629208:'#
!
_user_specified_name	2629206:'#
!
_user_specified_name	2629204:'#
!
_user_specified_name	2629202:WS
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
J
.__inference_MaskingLayer_layer_call_fn_2630393

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
I__inference_MaskingLayer_layer_call_and_return_conditional_losses_2628190d
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
�
y
]__inference_ReduceStackDimensionViaSummation_layer_call_and_return_conditional_losses_2631801

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
8__inference_transformer_encoder_27_layer_call_fn_2630482

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
S__inference_transformer_encoder_27_layer_call_and_return_conditional_losses_2629201s
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
_user_specified_name	2630476:'#
!
_user_specified_name	2630474:'#
!
_user_specified_name	2630472:'#
!
_user_specified_name	2630470:'#
!
_user_specified_name	2630468:'#
!
_user_specified_name	2630466:'
#
!
_user_specified_name	2630464:'	#
!
_user_specified_name	2630462:'#
!
_user_specified_name	2630460:'#
!
_user_specified_name	2630458:'#
!
_user_specified_name	2630456:'#
!
_user_specified_name	2630454:'#
!
_user_specified_name	2630452:'#
!
_user_specified_name	2630450:'#
!
_user_specified_name	2630448:'#
!
_user_specified_name	2630446:S O
+
_output_shapes
:���������P	
 
_user_specified_nameinputs
�
�
8__inference_transformer_encoder_29_layer_call_fn_2631362

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
S__inference_transformer_encoder_29_layer_call_and_return_conditional_losses_2629617s
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
_user_specified_name	2631356:'#
!
_user_specified_name	2631354:'#
!
_user_specified_name	2631352:'#
!
_user_specified_name	2631350:'#
!
_user_specified_name	2631348:'#
!
_user_specified_name	2631346:'
#
!
_user_specified_name	2631344:'	#
!
_user_specified_name	2631342:'#
!
_user_specified_name	2631340:'#
!
_user_specified_name	2631338:'#
!
_user_specified_name	2631336:'#
!
_user_specified_name	2631334:'#
!
_user_specified_name	2631332:'#
!
_user_specified_name	2631330:'#
!
_user_specified_name	2631328:'#
!
_user_specified_name	2631326:S O
+
_output_shapes
:���������P	
 
_user_specified_nameinputs
�
m
Q__inference_StandardizeTimeLimit_layer_call_and_return_conditional_losses_2631819

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
�q
�
D__inference_model_9_layer_call_and_return_conditional_losses_2629022
stacklevelinputfeatures
timelimitinput,
transformer_encoder_27_2628381:	,
transformer_encoder_27_2628383:	4
transformer_encoder_27_2628385:	0
transformer_encoder_27_2628387:4
transformer_encoder_27_2628389:	0
transformer_encoder_27_2628391:4
transformer_encoder_27_2628393:	0
transformer_encoder_27_2628395:4
transformer_encoder_27_2628397:	,
transformer_encoder_27_2628399:	,
transformer_encoder_27_2628401:	,
transformer_encoder_27_2628403:	0
transformer_encoder_27_2628405:		,
transformer_encoder_27_2628407:	0
transformer_encoder_27_2628409:		,
transformer_encoder_27_2628411:	,
transformer_encoder_28_2628603:	,
transformer_encoder_28_2628605:	4
transformer_encoder_28_2628607:	0
transformer_encoder_28_2628609:4
transformer_encoder_28_2628611:	0
transformer_encoder_28_2628613:4
transformer_encoder_28_2628615:	0
transformer_encoder_28_2628617:4
transformer_encoder_28_2628619:	,
transformer_encoder_28_2628621:	,
transformer_encoder_28_2628623:	,
transformer_encoder_28_2628625:	0
transformer_encoder_28_2628627:		,
transformer_encoder_28_2628629:	0
transformer_encoder_28_2628631:		,
transformer_encoder_28_2628633:	,
transformer_encoder_29_2628825:	,
transformer_encoder_29_2628827:	4
transformer_encoder_29_2628829:	0
transformer_encoder_29_2628831:4
transformer_encoder_29_2628833:	0
transformer_encoder_29_2628835:4
transformer_encoder_29_2628837:	0
transformer_encoder_29_2628839:4
transformer_encoder_29_2628841:	,
transformer_encoder_29_2628843:	,
transformer_encoder_29_2628845:	,
transformer_encoder_29_2628847:	0
transformer_encoder_29_2628849:		,
transformer_encoder_29_2628851:	0
transformer_encoder_29_2628853:		,
transformer_encoder_29_2628855:	$
finallayernorm_2628901:	$
finallayernorm_2628903:	6
$fullyconnectedlayerarearatio_2628951:

2
$fullyconnectedlayerarearatio_2628953:
-
predictionarearatio_2628967:
)
predictionarearatio_2628969:*
predictionstacks_2629003:	&
predictionstacks_2629005:
identity

identity_1��&FinalLayerNorm/StatefulPartitionedCall�4FullyConnectedLayerAreaRatio/StatefulPartitionedCall�+PredictionAreaRatio/StatefulPartitionedCall�(PredictionStacks/StatefulPartitionedCall�.transformer_encoder_27/StatefulPartitionedCall�.transformer_encoder_28/StatefulPartitionedCall�.transformer_encoder_29/StatefulPartitionedCall�
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
I__inference_MaskingLayer_layer_call_and_return_conditional_losses_2628190�
transformer_encoder_27/CastCast%MaskingLayer/PartitionedCall:output:0*

DstT0*

SrcT0*+
_output_shapes
:���������P	�
.transformer_encoder_27/StatefulPartitionedCallStatefulPartitionedCalltransformer_encoder_27/Cast:y:0transformer_encoder_27_2628381transformer_encoder_27_2628383transformer_encoder_27_2628385transformer_encoder_27_2628387transformer_encoder_27_2628389transformer_encoder_27_2628391transformer_encoder_27_2628393transformer_encoder_27_2628395transformer_encoder_27_2628397transformer_encoder_27_2628399transformer_encoder_27_2628401transformer_encoder_27_2628403transformer_encoder_27_2628405transformer_encoder_27_2628407transformer_encoder_27_2628409transformer_encoder_27_2628411*
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
S__inference_transformer_encoder_27_layer_call_and_return_conditional_losses_2628380�
.transformer_encoder_28/StatefulPartitionedCallStatefulPartitionedCall7transformer_encoder_27/StatefulPartitionedCall:output:0transformer_encoder_28_2628603transformer_encoder_28_2628605transformer_encoder_28_2628607transformer_encoder_28_2628609transformer_encoder_28_2628611transformer_encoder_28_2628613transformer_encoder_28_2628615transformer_encoder_28_2628617transformer_encoder_28_2628619transformer_encoder_28_2628621transformer_encoder_28_2628623transformer_encoder_28_2628625transformer_encoder_28_2628627transformer_encoder_28_2628629transformer_encoder_28_2628631transformer_encoder_28_2628633*
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
S__inference_transformer_encoder_28_layer_call_and_return_conditional_losses_2628602�
.transformer_encoder_29/StatefulPartitionedCallStatefulPartitionedCall7transformer_encoder_28/StatefulPartitionedCall:output:0transformer_encoder_29_2628825transformer_encoder_29_2628827transformer_encoder_29_2628829transformer_encoder_29_2628831transformer_encoder_29_2628833transformer_encoder_29_2628835transformer_encoder_29_2628837transformer_encoder_29_2628839transformer_encoder_29_2628841transformer_encoder_29_2628843transformer_encoder_29_2628845transformer_encoder_29_2628847transformer_encoder_29_2628849transformer_encoder_29_2628851transformer_encoder_29_2628853transformer_encoder_29_2628855*
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
S__inference_transformer_encoder_29_layer_call_and_return_conditional_losses_2628824�
&FinalLayerNorm/StatefulPartitionedCallStatefulPartitionedCall7transformer_encoder_29/StatefulPartitionedCall:output:0finallayernorm_2628901finallayernorm_2628903*
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
K__inference_FinalLayerNorm_layer_call_and_return_conditional_losses_2628900�
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
]__inference_ReduceStackDimensionViaSummation_layer_call_and_return_conditional_losses_2628913r
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
Q__inference_StandardizeTimeLimit_layer_call_and_return_conditional_losses_2628923�
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
M__inference_ConcatenateLayer_layer_call_and_return_conditional_losses_2628931�
4FullyConnectedLayerAreaRatio/StatefulPartitionedCallStatefulPartitionedCall)ConcatenateLayer/PartitionedCall:output:0$fullyconnectedlayerarearatio_2628951$fullyconnectedlayerarearatio_2628953*
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
Y__inference_FullyConnectedLayerAreaRatio_layer_call_and_return_conditional_losses_2628950�
+PredictionAreaRatio/StatefulPartitionedCallStatefulPartitionedCall=FullyConnectedLayerAreaRatio/StatefulPartitionedCall:output:0predictionarearatio_2628967predictionarearatio_2628969*
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
P__inference_PredictionAreaRatio_layer_call_and_return_conditional_losses_2628966�
(PredictionStacks/StatefulPartitionedCallStatefulPartitionedCall/FinalLayerNorm/StatefulPartitionedCall:output:0predictionstacks_2629003predictionstacks_2629005*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������P*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *V
fQRO
M__inference_PredictionStacks_layer_call_and_return_conditional_losses_2629002�
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
C__inference_Output_layer_call_and_return_conditional_losses_2629012�
Output/PartitionedCall_1PartitionedCall1PredictionStacks/StatefulPartitionedCall:output:0*
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
C__inference_Output_layer_call_and_return_conditional_losses_2629018a
IdentityIdentity!Output/PartitionedCall_1:output:0^NoOp*
T0*
_output_shapes
:a

Identity_1IdentityOutput/PartitionedCall:output:0^NoOp*
T0*
_output_shapes
:�
NoOpNoOp'^FinalLayerNorm/StatefulPartitionedCall5^FullyConnectedLayerAreaRatio/StatefulPartitionedCall,^PredictionAreaRatio/StatefulPartitionedCall)^PredictionStacks/StatefulPartitionedCall/^transformer_encoder_27/StatefulPartitionedCall/^transformer_encoder_28/StatefulPartitionedCall/^transformer_encoder_29/StatefulPartitionedCall*
_output_shapes
 "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:���������P	:���������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2P
&FinalLayerNorm/StatefulPartitionedCall&FinalLayerNorm/StatefulPartitionedCall2l
4FullyConnectedLayerAreaRatio/StatefulPartitionedCall4FullyConnectedLayerAreaRatio/StatefulPartitionedCall2Z
+PredictionAreaRatio/StatefulPartitionedCall+PredictionAreaRatio/StatefulPartitionedCall2T
(PredictionStacks/StatefulPartitionedCall(PredictionStacks/StatefulPartitionedCall2`
.transformer_encoder_27/StatefulPartitionedCall.transformer_encoder_27/StatefulPartitionedCall2`
.transformer_encoder_28/StatefulPartitionedCall.transformer_encoder_28/StatefulPartitionedCall2`
.transformer_encoder_29/StatefulPartitionedCall.transformer_encoder_29/StatefulPartitionedCall:'9#
!
_user_specified_name	2629005:'8#
!
_user_specified_name	2629003:'7#
!
_user_specified_name	2628969:'6#
!
_user_specified_name	2628967:'5#
!
_user_specified_name	2628953:'4#
!
_user_specified_name	2628951:'3#
!
_user_specified_name	2628903:'2#
!
_user_specified_name	2628901:'1#
!
_user_specified_name	2628855:'0#
!
_user_specified_name	2628853:'/#
!
_user_specified_name	2628851:'.#
!
_user_specified_name	2628849:'-#
!
_user_specified_name	2628847:',#
!
_user_specified_name	2628845:'+#
!
_user_specified_name	2628843:'*#
!
_user_specified_name	2628841:')#
!
_user_specified_name	2628839:'(#
!
_user_specified_name	2628837:''#
!
_user_specified_name	2628835:'&#
!
_user_specified_name	2628833:'%#
!
_user_specified_name	2628831:'$#
!
_user_specified_name	2628829:'##
!
_user_specified_name	2628827:'"#
!
_user_specified_name	2628825:'!#
!
_user_specified_name	2628633:' #
!
_user_specified_name	2628631:'#
!
_user_specified_name	2628629:'#
!
_user_specified_name	2628627:'#
!
_user_specified_name	2628625:'#
!
_user_specified_name	2628623:'#
!
_user_specified_name	2628621:'#
!
_user_specified_name	2628619:'#
!
_user_specified_name	2628617:'#
!
_user_specified_name	2628615:'#
!
_user_specified_name	2628613:'#
!
_user_specified_name	2628611:'#
!
_user_specified_name	2628609:'#
!
_user_specified_name	2628607:'#
!
_user_specified_name	2628605:'#
!
_user_specified_name	2628603:'#
!
_user_specified_name	2628411:'#
!
_user_specified_name	2628409:'#
!
_user_specified_name	2628407:'#
!
_user_specified_name	2628405:'#
!
_user_specified_name	2628403:'#
!
_user_specified_name	2628401:'#
!
_user_specified_name	2628399:'
#
!
_user_specified_name	2628397:'	#
!
_user_specified_name	2628395:'#
!
_user_specified_name	2628393:'#
!
_user_specified_name	2628391:'#
!
_user_specified_name	2628389:'#
!
_user_specified_name	2628387:'#
!
_user_specified_name	2628385:'#
!
_user_specified_name	2628383:'#
!
_user_specified_name	2628381:WS
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
S__inference_transformer_encoder_29_layer_call_and_return_conditional_losses_2628824

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
dropout_29/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?�
dropout_29/dropout/MulMul-Encoder-FeedForwardLayer_2_3/BiasAdd:output:0!dropout_29/dropout/Const:output:0*
T0*+
_output_shapes
:���������P	�
dropout_29/dropout/ShapeShape-Encoder-FeedForwardLayer_2_3/BiasAdd:output:0*
T0*
_output_shapes
::���
/dropout_29/dropout/random_uniform/RandomUniformRandomUniform!dropout_29/dropout/Shape:output:0*
T0*+
_output_shapes
:���������P	*
dtype0*
seed2*
seed��f
!dropout_29/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=�
dropout_29/dropout/GreaterEqualGreaterEqual8dropout_29/dropout/random_uniform/RandomUniform:output:0*dropout_29/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:���������P	_
dropout_29/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout_29/dropout/SelectV2SelectV2#dropout_29/dropout/GreaterEqual:z:0dropout_29/dropout/Mul:z:0#dropout_29/dropout/Const_1:output:0*
T0*+
_output_shapes
:���������P	�
Encoder-2nd-AdditionLayer-3/addAddV2#Encoder-1st-AdditionLayer-3/add:z:0$dropout_29/dropout/SelectV2:output:0*
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
C__inference_Output_layer_call_and_return_conditional_losses_2631962

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
S__inference_transformer_encoder_27_layer_call_and_return_conditional_losses_2628380

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
dropout_27/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?�
dropout_27/dropout/MulMul-Encoder-FeedForwardLayer_2_1/BiasAdd:output:0!dropout_27/dropout/Const:output:0*
T0*+
_output_shapes
:���������P	�
dropout_27/dropout/ShapeShape-Encoder-FeedForwardLayer_2_1/BiasAdd:output:0*
T0*
_output_shapes
::���
/dropout_27/dropout/random_uniform/RandomUniformRandomUniform!dropout_27/dropout/Shape:output:0*
T0*+
_output_shapes
:���������P	*
dtype0*
seed2*
seed��f
!dropout_27/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=�
dropout_27/dropout/GreaterEqualGreaterEqual8dropout_27/dropout/random_uniform/RandomUniform:output:0*dropout_27/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:���������P	_
dropout_27/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout_27/dropout/SelectV2SelectV2#dropout_27/dropout/GreaterEqual:z:0dropout_27/dropout/Mul:z:0#dropout_27/dropout/Const_1:output:0*
T0*+
_output_shapes
:���������P	�
Encoder-2nd-AdditionLayer-1/addAddV2#Encoder-1st-AdditionLayer-1/add:z:0$dropout_27/dropout/SelectV2:output:0*
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
�
y
]__inference_ReduceStackDimensionViaSummation_layer_call_and_return_conditional_losses_2631793

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
�
�
Y__inference_FullyConnectedLayerAreaRatio_layer_call_and_return_conditional_losses_2628950

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
��
�D
 __inference__traced_save_2632327
file_prefix9
+read_disablecopyonread_finallayernorm_gamma:	:
,read_1_disablecopyonread_finallayernorm_beta:	N
<read_2_disablecopyonread_fullyconnectedlayerarearatio_kernel:

H
:read_3_disablecopyonread_fullyconnectedlayerarearatio_bias:
B
0read_4_disablecopyonread_predictionstacks_kernel:	<
.read_5_disablecopyonread_predictionstacks_bias:E
3read_6_disablecopyonread_predictionarearatio_kernel:
?
1read_7_disablecopyonread_predictionarearatio_bias:o
Yread_8_disablecopyonread_transformer_encoder_27_encoder_selfattentionlayer_1_query_kernel:	i
Wread_9_disablecopyonread_transformer_encoder_27_encoder_selfattentionlayer_1_query_bias:n
Xread_10_disablecopyonread_transformer_encoder_27_encoder_selfattentionlayer_1_key_kernel:	h
Vread_11_disablecopyonread_transformer_encoder_27_encoder_selfattentionlayer_1_key_bias:p
Zread_12_disablecopyonread_transformer_encoder_27_encoder_selfattentionlayer_1_value_kernel:	j
Xread_13_disablecopyonread_transformer_encoder_27_encoder_selfattentionlayer_1_value_bias:{
eread_14_disablecopyonread_transformer_encoder_27_encoder_selfattentionlayer_1_attention_output_kernel:	q
cread_15_disablecopyonread_transformer_encoder_27_encoder_selfattentionlayer_1_attention_output_bias:	e
Wread_16_disablecopyonread_transformer_encoder_27_encoder_1st_normalizationlayer_1_gamma:	d
Vread_17_disablecopyonread_transformer_encoder_27_encoder_1st_normalizationlayer_1_beta:	e
Wread_18_disablecopyonread_transformer_encoder_27_encoder_2nd_normalizationlayer_1_gamma:	d
Vread_19_disablecopyonread_transformer_encoder_27_encoder_2nd_normalizationlayer_1_beta:	f
Tread_20_disablecopyonread_transformer_encoder_27_encoder_feedforwardlayer_1_1_kernel:		`
Rread_21_disablecopyonread_transformer_encoder_27_encoder_feedforwardlayer_1_1_bias:	f
Tread_22_disablecopyonread_transformer_encoder_27_encoder_feedforwardlayer_2_1_kernel:		`
Rread_23_disablecopyonread_transformer_encoder_27_encoder_feedforwardlayer_2_1_bias:	p
Zread_24_disablecopyonread_transformer_encoder_28_encoder_selfattentionlayer_2_query_kernel:	j
Xread_25_disablecopyonread_transformer_encoder_28_encoder_selfattentionlayer_2_query_bias:n
Xread_26_disablecopyonread_transformer_encoder_28_encoder_selfattentionlayer_2_key_kernel:	h
Vread_27_disablecopyonread_transformer_encoder_28_encoder_selfattentionlayer_2_key_bias:p
Zread_28_disablecopyonread_transformer_encoder_28_encoder_selfattentionlayer_2_value_kernel:	j
Xread_29_disablecopyonread_transformer_encoder_28_encoder_selfattentionlayer_2_value_bias:{
eread_30_disablecopyonread_transformer_encoder_28_encoder_selfattentionlayer_2_attention_output_kernel:	q
cread_31_disablecopyonread_transformer_encoder_28_encoder_selfattentionlayer_2_attention_output_bias:	e
Wread_32_disablecopyonread_transformer_encoder_28_encoder_1st_normalizationlayer_2_gamma:	d
Vread_33_disablecopyonread_transformer_encoder_28_encoder_1st_normalizationlayer_2_beta:	e
Wread_34_disablecopyonread_transformer_encoder_28_encoder_2nd_normalizationlayer_2_gamma:	d
Vread_35_disablecopyonread_transformer_encoder_28_encoder_2nd_normalizationlayer_2_beta:	f
Tread_36_disablecopyonread_transformer_encoder_28_encoder_feedforwardlayer_1_2_kernel:		`
Rread_37_disablecopyonread_transformer_encoder_28_encoder_feedforwardlayer_1_2_bias:	f
Tread_38_disablecopyonread_transformer_encoder_28_encoder_feedforwardlayer_2_2_kernel:		`
Rread_39_disablecopyonread_transformer_encoder_28_encoder_feedforwardlayer_2_2_bias:	p
Zread_40_disablecopyonread_transformer_encoder_29_encoder_selfattentionlayer_3_query_kernel:	j
Xread_41_disablecopyonread_transformer_encoder_29_encoder_selfattentionlayer_3_query_bias:n
Xread_42_disablecopyonread_transformer_encoder_29_encoder_selfattentionlayer_3_key_kernel:	h
Vread_43_disablecopyonread_transformer_encoder_29_encoder_selfattentionlayer_3_key_bias:p
Zread_44_disablecopyonread_transformer_encoder_29_encoder_selfattentionlayer_3_value_kernel:	j
Xread_45_disablecopyonread_transformer_encoder_29_encoder_selfattentionlayer_3_value_bias:{
eread_46_disablecopyonread_transformer_encoder_29_encoder_selfattentionlayer_3_attention_output_kernel:	q
cread_47_disablecopyonread_transformer_encoder_29_encoder_selfattentionlayer_3_attention_output_bias:	e
Wread_48_disablecopyonread_transformer_encoder_29_encoder_1st_normalizationlayer_3_gamma:	d
Vread_49_disablecopyonread_transformer_encoder_29_encoder_1st_normalizationlayer_3_beta:	e
Wread_50_disablecopyonread_transformer_encoder_29_encoder_2nd_normalizationlayer_3_gamma:	d
Vread_51_disablecopyonread_transformer_encoder_29_encoder_2nd_normalizationlayer_3_beta:	f
Tread_52_disablecopyonread_transformer_encoder_29_encoder_feedforwardlayer_1_3_kernel:		`
Rread_53_disablecopyonread_transformer_encoder_29_encoder_feedforwardlayer_1_3_bias:	f
Tread_54_disablecopyonread_transformer_encoder_29_encoder_feedforwardlayer_2_3_kernel:		`
Rread_55_disablecopyonread_transformer_encoder_29_encoder_feedforwardlayer_2_3_bias:	
savev2_const
identity_113��MergeV2Checkpoints�Read/DisableCopyOnRead�Read/ReadVariableOp�Read_1/DisableCopyOnRead�Read_1/ReadVariableOp�Read_10/DisableCopyOnRead�Read_10/ReadVariableOp�Read_11/DisableCopyOnRead�Read_11/ReadVariableOp�Read_12/DisableCopyOnRead�Read_12/ReadVariableOp�Read_13/DisableCopyOnRead�Read_13/ReadVariableOp�Read_14/DisableCopyOnRead�Read_14/ReadVariableOp�Read_15/DisableCopyOnRead�Read_15/ReadVariableOp�Read_16/DisableCopyOnRead�Read_16/ReadVariableOp�Read_17/DisableCopyOnRead�Read_17/ReadVariableOp�Read_18/DisableCopyOnRead�Read_18/ReadVariableOp�Read_19/DisableCopyOnRead�Read_19/ReadVariableOp�Read_2/DisableCopyOnRead�Read_2/ReadVariableOp�Read_20/DisableCopyOnRead�Read_20/ReadVariableOp�Read_21/DisableCopyOnRead�Read_21/ReadVariableOp�Read_22/DisableCopyOnRead�Read_22/ReadVariableOp�Read_23/DisableCopyOnRead�Read_23/ReadVariableOp�Read_24/DisableCopyOnRead�Read_24/ReadVariableOp�Read_25/DisableCopyOnRead�Read_25/ReadVariableOp�Read_26/DisableCopyOnRead�Read_26/ReadVariableOp�Read_27/DisableCopyOnRead�Read_27/ReadVariableOp�Read_28/DisableCopyOnRead�Read_28/ReadVariableOp�Read_29/DisableCopyOnRead�Read_29/ReadVariableOp�Read_3/DisableCopyOnRead�Read_3/ReadVariableOp�Read_30/DisableCopyOnRead�Read_30/ReadVariableOp�Read_31/DisableCopyOnRead�Read_31/ReadVariableOp�Read_32/DisableCopyOnRead�Read_32/ReadVariableOp�Read_33/DisableCopyOnRead�Read_33/ReadVariableOp�Read_34/DisableCopyOnRead�Read_34/ReadVariableOp�Read_35/DisableCopyOnRead�Read_35/ReadVariableOp�Read_36/DisableCopyOnRead�Read_36/ReadVariableOp�Read_37/DisableCopyOnRead�Read_37/ReadVariableOp�Read_38/DisableCopyOnRead�Read_38/ReadVariableOp�Read_39/DisableCopyOnRead�Read_39/ReadVariableOp�Read_4/DisableCopyOnRead�Read_4/ReadVariableOp�Read_40/DisableCopyOnRead�Read_40/ReadVariableOp�Read_41/DisableCopyOnRead�Read_41/ReadVariableOp�Read_42/DisableCopyOnRead�Read_42/ReadVariableOp�Read_43/DisableCopyOnRead�Read_43/ReadVariableOp�Read_44/DisableCopyOnRead�Read_44/ReadVariableOp�Read_45/DisableCopyOnRead�Read_45/ReadVariableOp�Read_46/DisableCopyOnRead�Read_46/ReadVariableOp�Read_47/DisableCopyOnRead�Read_47/ReadVariableOp�Read_48/DisableCopyOnRead�Read_48/ReadVariableOp�Read_49/DisableCopyOnRead�Read_49/ReadVariableOp�Read_5/DisableCopyOnRead�Read_5/ReadVariableOp�Read_50/DisableCopyOnRead�Read_50/ReadVariableOp�Read_51/DisableCopyOnRead�Read_51/ReadVariableOp�Read_52/DisableCopyOnRead�Read_52/ReadVariableOp�Read_53/DisableCopyOnRead�Read_53/ReadVariableOp�Read_54/DisableCopyOnRead�Read_54/ReadVariableOp�Read_55/DisableCopyOnRead�Read_55/ReadVariableOp�Read_6/DisableCopyOnRead�Read_6/ReadVariableOp�Read_7/DisableCopyOnRead�Read_7/ReadVariableOp�Read_8/DisableCopyOnRead�Read_8/ReadVariableOp�Read_9/DisableCopyOnRead�Read_9/ReadVariableOpw
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
Read_4/DisableCopyOnReadDisableCopyOnRead0read_4_disablecopyonread_predictionstacks_kernel"/device:CPU:0*
_output_shapes
 �
Read_4/ReadVariableOpReadVariableOp0read_4_disablecopyonread_predictionstacks_kernel^Read_4/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:	*
dtype0m

Identity_8IdentityRead_4/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:	c

Identity_9IdentityIdentity_8:output:0"/device:CPU:0*
T0*
_output_shapes

:	�
Read_5/DisableCopyOnReadDisableCopyOnRead.read_5_disablecopyonread_predictionstacks_bias"/device:CPU:0*
_output_shapes
 �
Read_5/ReadVariableOpReadVariableOp.read_5_disablecopyonread_predictionstacks_bias^Read_5/DisableCopyOnRead"/device:CPU:0*
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
Read_8/DisableCopyOnReadDisableCopyOnReadYread_8_disablecopyonread_transformer_encoder_27_encoder_selfattentionlayer_1_query_kernel"/device:CPU:0*
_output_shapes
 �
Read_8/ReadVariableOpReadVariableOpYread_8_disablecopyonread_transformer_encoder_27_encoder_selfattentionlayer_1_query_kernel^Read_8/DisableCopyOnRead"/device:CPU:0*"
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
Read_9/DisableCopyOnReadDisableCopyOnReadWread_9_disablecopyonread_transformer_encoder_27_encoder_selfattentionlayer_1_query_bias"/device:CPU:0*
_output_shapes
 �
Read_9/ReadVariableOpReadVariableOpWread_9_disablecopyonread_transformer_encoder_27_encoder_selfattentionlayer_1_query_bias^Read_9/DisableCopyOnRead"/device:CPU:0*
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
Read_10/DisableCopyOnReadDisableCopyOnReadXread_10_disablecopyonread_transformer_encoder_27_encoder_selfattentionlayer_1_key_kernel"/device:CPU:0*
_output_shapes
 �
Read_10/ReadVariableOpReadVariableOpXread_10_disablecopyonread_transformer_encoder_27_encoder_selfattentionlayer_1_key_kernel^Read_10/DisableCopyOnRead"/device:CPU:0*"
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
Read_11/DisableCopyOnReadDisableCopyOnReadVread_11_disablecopyonread_transformer_encoder_27_encoder_selfattentionlayer_1_key_bias"/device:CPU:0*
_output_shapes
 �
Read_11/ReadVariableOpReadVariableOpVread_11_disablecopyonread_transformer_encoder_27_encoder_selfattentionlayer_1_key_bias^Read_11/DisableCopyOnRead"/device:CPU:0*
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
Read_12/DisableCopyOnReadDisableCopyOnReadZread_12_disablecopyonread_transformer_encoder_27_encoder_selfattentionlayer_1_value_kernel"/device:CPU:0*
_output_shapes
 �
Read_12/ReadVariableOpReadVariableOpZread_12_disablecopyonread_transformer_encoder_27_encoder_selfattentionlayer_1_value_kernel^Read_12/DisableCopyOnRead"/device:CPU:0*"
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
Read_13/DisableCopyOnReadDisableCopyOnReadXread_13_disablecopyonread_transformer_encoder_27_encoder_selfattentionlayer_1_value_bias"/device:CPU:0*
_output_shapes
 �
Read_13/ReadVariableOpReadVariableOpXread_13_disablecopyonread_transformer_encoder_27_encoder_selfattentionlayer_1_value_bias^Read_13/DisableCopyOnRead"/device:CPU:0*
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
Read_14/DisableCopyOnReadDisableCopyOnReaderead_14_disablecopyonread_transformer_encoder_27_encoder_selfattentionlayer_1_attention_output_kernel"/device:CPU:0*
_output_shapes
 �
Read_14/ReadVariableOpReadVariableOperead_14_disablecopyonread_transformer_encoder_27_encoder_selfattentionlayer_1_attention_output_kernel^Read_14/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
:	*
dtype0s
Identity_28IdentityRead_14/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
:	i
Identity_29IdentityIdentity_28:output:0"/device:CPU:0*
T0*"
_output_shapes
:	�
Read_15/DisableCopyOnReadDisableCopyOnReadcread_15_disablecopyonread_transformer_encoder_27_encoder_selfattentionlayer_1_attention_output_bias"/device:CPU:0*
_output_shapes
 �
Read_15/ReadVariableOpReadVariableOpcread_15_disablecopyonread_transformer_encoder_27_encoder_selfattentionlayer_1_attention_output_bias^Read_15/DisableCopyOnRead"/device:CPU:0*
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
Read_16/DisableCopyOnReadDisableCopyOnReadWread_16_disablecopyonread_transformer_encoder_27_encoder_1st_normalizationlayer_1_gamma"/device:CPU:0*
_output_shapes
 �
Read_16/ReadVariableOpReadVariableOpWread_16_disablecopyonread_transformer_encoder_27_encoder_1st_normalizationlayer_1_gamma^Read_16/DisableCopyOnRead"/device:CPU:0*
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
Read_17/DisableCopyOnReadDisableCopyOnReadVread_17_disablecopyonread_transformer_encoder_27_encoder_1st_normalizationlayer_1_beta"/device:CPU:0*
_output_shapes
 �
Read_17/ReadVariableOpReadVariableOpVread_17_disablecopyonread_transformer_encoder_27_encoder_1st_normalizationlayer_1_beta^Read_17/DisableCopyOnRead"/device:CPU:0*
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
Read_18/DisableCopyOnReadDisableCopyOnReadWread_18_disablecopyonread_transformer_encoder_27_encoder_2nd_normalizationlayer_1_gamma"/device:CPU:0*
_output_shapes
 �
Read_18/ReadVariableOpReadVariableOpWread_18_disablecopyonread_transformer_encoder_27_encoder_2nd_normalizationlayer_1_gamma^Read_18/DisableCopyOnRead"/device:CPU:0*
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
Read_19/DisableCopyOnReadDisableCopyOnReadVread_19_disablecopyonread_transformer_encoder_27_encoder_2nd_normalizationlayer_1_beta"/device:CPU:0*
_output_shapes
 �
Read_19/ReadVariableOpReadVariableOpVread_19_disablecopyonread_transformer_encoder_27_encoder_2nd_normalizationlayer_1_beta^Read_19/DisableCopyOnRead"/device:CPU:0*
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
Read_20/DisableCopyOnReadDisableCopyOnReadTread_20_disablecopyonread_transformer_encoder_27_encoder_feedforwardlayer_1_1_kernel"/device:CPU:0*
_output_shapes
 �
Read_20/ReadVariableOpReadVariableOpTread_20_disablecopyonread_transformer_encoder_27_encoder_feedforwardlayer_1_1_kernel^Read_20/DisableCopyOnRead"/device:CPU:0*
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
Read_21/DisableCopyOnReadDisableCopyOnReadRread_21_disablecopyonread_transformer_encoder_27_encoder_feedforwardlayer_1_1_bias"/device:CPU:0*
_output_shapes
 �
Read_21/ReadVariableOpReadVariableOpRread_21_disablecopyonread_transformer_encoder_27_encoder_feedforwardlayer_1_1_bias^Read_21/DisableCopyOnRead"/device:CPU:0*
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
Read_22/DisableCopyOnReadDisableCopyOnReadTread_22_disablecopyonread_transformer_encoder_27_encoder_feedforwardlayer_2_1_kernel"/device:CPU:0*
_output_shapes
 �
Read_22/ReadVariableOpReadVariableOpTread_22_disablecopyonread_transformer_encoder_27_encoder_feedforwardlayer_2_1_kernel^Read_22/DisableCopyOnRead"/device:CPU:0*
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
Read_23/DisableCopyOnReadDisableCopyOnReadRread_23_disablecopyonread_transformer_encoder_27_encoder_feedforwardlayer_2_1_bias"/device:CPU:0*
_output_shapes
 �
Read_23/ReadVariableOpReadVariableOpRread_23_disablecopyonread_transformer_encoder_27_encoder_feedforwardlayer_2_1_bias^Read_23/DisableCopyOnRead"/device:CPU:0*
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
Read_24/DisableCopyOnReadDisableCopyOnReadZread_24_disablecopyonread_transformer_encoder_28_encoder_selfattentionlayer_2_query_kernel"/device:CPU:0*
_output_shapes
 �
Read_24/ReadVariableOpReadVariableOpZread_24_disablecopyonread_transformer_encoder_28_encoder_selfattentionlayer_2_query_kernel^Read_24/DisableCopyOnRead"/device:CPU:0*"
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
Read_25/DisableCopyOnReadDisableCopyOnReadXread_25_disablecopyonread_transformer_encoder_28_encoder_selfattentionlayer_2_query_bias"/device:CPU:0*
_output_shapes
 �
Read_25/ReadVariableOpReadVariableOpXread_25_disablecopyonread_transformer_encoder_28_encoder_selfattentionlayer_2_query_bias^Read_25/DisableCopyOnRead"/device:CPU:0*
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
Read_26/DisableCopyOnReadDisableCopyOnReadXread_26_disablecopyonread_transformer_encoder_28_encoder_selfattentionlayer_2_key_kernel"/device:CPU:0*
_output_shapes
 �
Read_26/ReadVariableOpReadVariableOpXread_26_disablecopyonread_transformer_encoder_28_encoder_selfattentionlayer_2_key_kernel^Read_26/DisableCopyOnRead"/device:CPU:0*"
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
Read_27/DisableCopyOnReadDisableCopyOnReadVread_27_disablecopyonread_transformer_encoder_28_encoder_selfattentionlayer_2_key_bias"/device:CPU:0*
_output_shapes
 �
Read_27/ReadVariableOpReadVariableOpVread_27_disablecopyonread_transformer_encoder_28_encoder_selfattentionlayer_2_key_bias^Read_27/DisableCopyOnRead"/device:CPU:0*
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
Read_28/DisableCopyOnReadDisableCopyOnReadZread_28_disablecopyonread_transformer_encoder_28_encoder_selfattentionlayer_2_value_kernel"/device:CPU:0*
_output_shapes
 �
Read_28/ReadVariableOpReadVariableOpZread_28_disablecopyonread_transformer_encoder_28_encoder_selfattentionlayer_2_value_kernel^Read_28/DisableCopyOnRead"/device:CPU:0*"
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
Read_29/DisableCopyOnReadDisableCopyOnReadXread_29_disablecopyonread_transformer_encoder_28_encoder_selfattentionlayer_2_value_bias"/device:CPU:0*
_output_shapes
 �
Read_29/ReadVariableOpReadVariableOpXread_29_disablecopyonread_transformer_encoder_28_encoder_selfattentionlayer_2_value_bias^Read_29/DisableCopyOnRead"/device:CPU:0*
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
Read_30/DisableCopyOnReadDisableCopyOnReaderead_30_disablecopyonread_transformer_encoder_28_encoder_selfattentionlayer_2_attention_output_kernel"/device:CPU:0*
_output_shapes
 �
Read_30/ReadVariableOpReadVariableOperead_30_disablecopyonread_transformer_encoder_28_encoder_selfattentionlayer_2_attention_output_kernel^Read_30/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
:	*
dtype0s
Identity_60IdentityRead_30/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
:	i
Identity_61IdentityIdentity_60:output:0"/device:CPU:0*
T0*"
_output_shapes
:	�
Read_31/DisableCopyOnReadDisableCopyOnReadcread_31_disablecopyonread_transformer_encoder_28_encoder_selfattentionlayer_2_attention_output_bias"/device:CPU:0*
_output_shapes
 �
Read_31/ReadVariableOpReadVariableOpcread_31_disablecopyonread_transformer_encoder_28_encoder_selfattentionlayer_2_attention_output_bias^Read_31/DisableCopyOnRead"/device:CPU:0*
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
Read_32/DisableCopyOnReadDisableCopyOnReadWread_32_disablecopyonread_transformer_encoder_28_encoder_1st_normalizationlayer_2_gamma"/device:CPU:0*
_output_shapes
 �
Read_32/ReadVariableOpReadVariableOpWread_32_disablecopyonread_transformer_encoder_28_encoder_1st_normalizationlayer_2_gamma^Read_32/DisableCopyOnRead"/device:CPU:0*
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
Read_33/DisableCopyOnReadDisableCopyOnReadVread_33_disablecopyonread_transformer_encoder_28_encoder_1st_normalizationlayer_2_beta"/device:CPU:0*
_output_shapes
 �
Read_33/ReadVariableOpReadVariableOpVread_33_disablecopyonread_transformer_encoder_28_encoder_1st_normalizationlayer_2_beta^Read_33/DisableCopyOnRead"/device:CPU:0*
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
Read_34/DisableCopyOnReadDisableCopyOnReadWread_34_disablecopyonread_transformer_encoder_28_encoder_2nd_normalizationlayer_2_gamma"/device:CPU:0*
_output_shapes
 �
Read_34/ReadVariableOpReadVariableOpWread_34_disablecopyonread_transformer_encoder_28_encoder_2nd_normalizationlayer_2_gamma^Read_34/DisableCopyOnRead"/device:CPU:0*
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
Read_35/DisableCopyOnReadDisableCopyOnReadVread_35_disablecopyonread_transformer_encoder_28_encoder_2nd_normalizationlayer_2_beta"/device:CPU:0*
_output_shapes
 �
Read_35/ReadVariableOpReadVariableOpVread_35_disablecopyonread_transformer_encoder_28_encoder_2nd_normalizationlayer_2_beta^Read_35/DisableCopyOnRead"/device:CPU:0*
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
Read_36/DisableCopyOnReadDisableCopyOnReadTread_36_disablecopyonread_transformer_encoder_28_encoder_feedforwardlayer_1_2_kernel"/device:CPU:0*
_output_shapes
 �
Read_36/ReadVariableOpReadVariableOpTread_36_disablecopyonread_transformer_encoder_28_encoder_feedforwardlayer_1_2_kernel^Read_36/DisableCopyOnRead"/device:CPU:0*
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
Read_37/DisableCopyOnReadDisableCopyOnReadRread_37_disablecopyonread_transformer_encoder_28_encoder_feedforwardlayer_1_2_bias"/device:CPU:0*
_output_shapes
 �
Read_37/ReadVariableOpReadVariableOpRread_37_disablecopyonread_transformer_encoder_28_encoder_feedforwardlayer_1_2_bias^Read_37/DisableCopyOnRead"/device:CPU:0*
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
Read_38/DisableCopyOnReadDisableCopyOnReadTread_38_disablecopyonread_transformer_encoder_28_encoder_feedforwardlayer_2_2_kernel"/device:CPU:0*
_output_shapes
 �
Read_38/ReadVariableOpReadVariableOpTread_38_disablecopyonread_transformer_encoder_28_encoder_feedforwardlayer_2_2_kernel^Read_38/DisableCopyOnRead"/device:CPU:0*
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
Read_39/DisableCopyOnReadDisableCopyOnReadRread_39_disablecopyonread_transformer_encoder_28_encoder_feedforwardlayer_2_2_bias"/device:CPU:0*
_output_shapes
 �
Read_39/ReadVariableOpReadVariableOpRread_39_disablecopyonread_transformer_encoder_28_encoder_feedforwardlayer_2_2_bias^Read_39/DisableCopyOnRead"/device:CPU:0*
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
Read_40/DisableCopyOnReadDisableCopyOnReadZread_40_disablecopyonread_transformer_encoder_29_encoder_selfattentionlayer_3_query_kernel"/device:CPU:0*
_output_shapes
 �
Read_40/ReadVariableOpReadVariableOpZread_40_disablecopyonread_transformer_encoder_29_encoder_selfattentionlayer_3_query_kernel^Read_40/DisableCopyOnRead"/device:CPU:0*"
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
Read_41/DisableCopyOnReadDisableCopyOnReadXread_41_disablecopyonread_transformer_encoder_29_encoder_selfattentionlayer_3_query_bias"/device:CPU:0*
_output_shapes
 �
Read_41/ReadVariableOpReadVariableOpXread_41_disablecopyonread_transformer_encoder_29_encoder_selfattentionlayer_3_query_bias^Read_41/DisableCopyOnRead"/device:CPU:0*
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
Read_42/DisableCopyOnReadDisableCopyOnReadXread_42_disablecopyonread_transformer_encoder_29_encoder_selfattentionlayer_3_key_kernel"/device:CPU:0*
_output_shapes
 �
Read_42/ReadVariableOpReadVariableOpXread_42_disablecopyonread_transformer_encoder_29_encoder_selfattentionlayer_3_key_kernel^Read_42/DisableCopyOnRead"/device:CPU:0*"
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
Read_43/DisableCopyOnReadDisableCopyOnReadVread_43_disablecopyonread_transformer_encoder_29_encoder_selfattentionlayer_3_key_bias"/device:CPU:0*
_output_shapes
 �
Read_43/ReadVariableOpReadVariableOpVread_43_disablecopyonread_transformer_encoder_29_encoder_selfattentionlayer_3_key_bias^Read_43/DisableCopyOnRead"/device:CPU:0*
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
Read_44/DisableCopyOnReadDisableCopyOnReadZread_44_disablecopyonread_transformer_encoder_29_encoder_selfattentionlayer_3_value_kernel"/device:CPU:0*
_output_shapes
 �
Read_44/ReadVariableOpReadVariableOpZread_44_disablecopyonread_transformer_encoder_29_encoder_selfattentionlayer_3_value_kernel^Read_44/DisableCopyOnRead"/device:CPU:0*"
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
Read_45/DisableCopyOnReadDisableCopyOnReadXread_45_disablecopyonread_transformer_encoder_29_encoder_selfattentionlayer_3_value_bias"/device:CPU:0*
_output_shapes
 �
Read_45/ReadVariableOpReadVariableOpXread_45_disablecopyonread_transformer_encoder_29_encoder_selfattentionlayer_3_value_bias^Read_45/DisableCopyOnRead"/device:CPU:0*
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
Read_46/DisableCopyOnReadDisableCopyOnReaderead_46_disablecopyonread_transformer_encoder_29_encoder_selfattentionlayer_3_attention_output_kernel"/device:CPU:0*
_output_shapes
 �
Read_46/ReadVariableOpReadVariableOperead_46_disablecopyonread_transformer_encoder_29_encoder_selfattentionlayer_3_attention_output_kernel^Read_46/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
:	*
dtype0s
Identity_92IdentityRead_46/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
:	i
Identity_93IdentityIdentity_92:output:0"/device:CPU:0*
T0*"
_output_shapes
:	�
Read_47/DisableCopyOnReadDisableCopyOnReadcread_47_disablecopyonread_transformer_encoder_29_encoder_selfattentionlayer_3_attention_output_bias"/device:CPU:0*
_output_shapes
 �
Read_47/ReadVariableOpReadVariableOpcread_47_disablecopyonread_transformer_encoder_29_encoder_selfattentionlayer_3_attention_output_bias^Read_47/DisableCopyOnRead"/device:CPU:0*
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
Read_48/DisableCopyOnReadDisableCopyOnReadWread_48_disablecopyonread_transformer_encoder_29_encoder_1st_normalizationlayer_3_gamma"/device:CPU:0*
_output_shapes
 �
Read_48/ReadVariableOpReadVariableOpWread_48_disablecopyonread_transformer_encoder_29_encoder_1st_normalizationlayer_3_gamma^Read_48/DisableCopyOnRead"/device:CPU:0*
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
Read_49/DisableCopyOnReadDisableCopyOnReadVread_49_disablecopyonread_transformer_encoder_29_encoder_1st_normalizationlayer_3_beta"/device:CPU:0*
_output_shapes
 �
Read_49/ReadVariableOpReadVariableOpVread_49_disablecopyonread_transformer_encoder_29_encoder_1st_normalizationlayer_3_beta^Read_49/DisableCopyOnRead"/device:CPU:0*
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
Read_50/DisableCopyOnReadDisableCopyOnReadWread_50_disablecopyonread_transformer_encoder_29_encoder_2nd_normalizationlayer_3_gamma"/device:CPU:0*
_output_shapes
 �
Read_50/ReadVariableOpReadVariableOpWread_50_disablecopyonread_transformer_encoder_29_encoder_2nd_normalizationlayer_3_gamma^Read_50/DisableCopyOnRead"/device:CPU:0*
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
Read_51/DisableCopyOnReadDisableCopyOnReadVread_51_disablecopyonread_transformer_encoder_29_encoder_2nd_normalizationlayer_3_beta"/device:CPU:0*
_output_shapes
 �
Read_51/ReadVariableOpReadVariableOpVread_51_disablecopyonread_transformer_encoder_29_encoder_2nd_normalizationlayer_3_beta^Read_51/DisableCopyOnRead"/device:CPU:0*
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
Read_52/DisableCopyOnReadDisableCopyOnReadTread_52_disablecopyonread_transformer_encoder_29_encoder_feedforwardlayer_1_3_kernel"/device:CPU:0*
_output_shapes
 �
Read_52/ReadVariableOpReadVariableOpTread_52_disablecopyonread_transformer_encoder_29_encoder_feedforwardlayer_1_3_kernel^Read_52/DisableCopyOnRead"/device:CPU:0*
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
Read_53/DisableCopyOnReadDisableCopyOnReadRread_53_disablecopyonread_transformer_encoder_29_encoder_feedforwardlayer_1_3_bias"/device:CPU:0*
_output_shapes
 �
Read_53/ReadVariableOpReadVariableOpRread_53_disablecopyonread_transformer_encoder_29_encoder_feedforwardlayer_1_3_bias^Read_53/DisableCopyOnRead"/device:CPU:0*
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
Read_54/DisableCopyOnReadDisableCopyOnReadTread_54_disablecopyonread_transformer_encoder_29_encoder_feedforwardlayer_2_3_kernel"/device:CPU:0*
_output_shapes
 �
Read_54/ReadVariableOpReadVariableOpTread_54_disablecopyonread_transformer_encoder_29_encoder_feedforwardlayer_2_3_kernel^Read_54/DisableCopyOnRead"/device:CPU:0*
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
Read_55/DisableCopyOnReadDisableCopyOnReadRread_55_disablecopyonread_transformer_encoder_29_encoder_feedforwardlayer_2_3_bias"/device:CPU:0*
_output_shapes
 �
Read_55/ReadVariableOpReadVariableOpRread_55_disablecopyonread_transformer_encoder_29_encoder_feedforwardlayer_2_3_bias^Read_55/DisableCopyOnRead"/device:CPU:0*
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
:	�
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:9*
dtype0*�
value�B�9B5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB'variables/26/.ATTRIBUTES/VARIABLE_VALUEB'variables/27/.ATTRIBUTES/VARIABLE_VALUEB'variables/28/.ATTRIBUTES/VARIABLE_VALUEB'variables/29/.ATTRIBUTES/VARIABLE_VALUEB'variables/30/.ATTRIBUTES/VARIABLE_VALUEB'variables/31/.ATTRIBUTES/VARIABLE_VALUEB'variables/32/.ATTRIBUTES/VARIABLE_VALUEB'variables/33/.ATTRIBUTES/VARIABLE_VALUEB'variables/34/.ATTRIBUTES/VARIABLE_VALUEB'variables/35/.ATTRIBUTES/VARIABLE_VALUEB'variables/36/.ATTRIBUTES/VARIABLE_VALUEB'variables/37/.ATTRIBUTES/VARIABLE_VALUEB'variables/38/.ATTRIBUTES/VARIABLE_VALUEB'variables/39/.ATTRIBUTES/VARIABLE_VALUEB'variables/40/.ATTRIBUTES/VARIABLE_VALUEB'variables/41/.ATTRIBUTES/VARIABLE_VALUEB'variables/42/.ATTRIBUTES/VARIABLE_VALUEB'variables/43/.ATTRIBUTES/VARIABLE_VALUEB'variables/44/.ATTRIBUTES/VARIABLE_VALUEB'variables/45/.ATTRIBUTES/VARIABLE_VALUEB'variables/46/.ATTRIBUTES/VARIABLE_VALUEB'variables/47/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:9*
dtype0*�
value|Bz9B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Identity_1:output:0Identity_3:output:0Identity_5:output:0Identity_7:output:0Identity_9:output:0Identity_11:output:0Identity_13:output:0Identity_15:output:0Identity_17:output:0Identity_19:output:0Identity_21:output:0Identity_23:output:0Identity_25:output:0Identity_27:output:0Identity_29:output:0Identity_31:output:0Identity_33:output:0Identity_35:output:0Identity_37:output:0Identity_39:output:0Identity_41:output:0Identity_43:output:0Identity_45:output:0Identity_47:output:0Identity_49:output:0Identity_51:output:0Identity_53:output:0Identity_55:output:0Identity_57:output:0Identity_59:output:0Identity_61:output:0Identity_63:output:0Identity_65:output:0Identity_67:output:0Identity_69:output:0Identity_71:output:0Identity_73:output:0Identity_75:output:0Identity_77:output:0Identity_79:output:0Identity_81:output:0Identity_83:output:0Identity_85:output:0Identity_87:output:0Identity_89:output:0Identity_91:output:0Identity_93:output:0Identity_95:output:0Identity_97:output:0Identity_99:output:0Identity_101:output:0Identity_103:output:0Identity_105:output:0Identity_107:output:0Identity_109:output:0Identity_111:output:0savev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *G
dtypes=
;29�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 j
Identity_112Identityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: W
Identity_113IdentityIdentity_112:output:0^NoOp*
T0*
_output_shapes
: �
NoOpNoOp^MergeV2Checkpoints^Read/DisableCopyOnRead^Read/ReadVariableOp^Read_1/DisableCopyOnRead^Read_1/ReadVariableOp^Read_10/DisableCopyOnRead^Read_10/ReadVariableOp^Read_11/DisableCopyOnRead^Read_11/ReadVariableOp^Read_12/DisableCopyOnRead^Read_12/ReadVariableOp^Read_13/DisableCopyOnRead^Read_13/ReadVariableOp^Read_14/DisableCopyOnRead^Read_14/ReadVariableOp^Read_15/DisableCopyOnRead^Read_15/ReadVariableOp^Read_16/DisableCopyOnRead^Read_16/ReadVariableOp^Read_17/DisableCopyOnRead^Read_17/ReadVariableOp^Read_18/DisableCopyOnRead^Read_18/ReadVariableOp^Read_19/DisableCopyOnRead^Read_19/ReadVariableOp^Read_2/DisableCopyOnRead^Read_2/ReadVariableOp^Read_20/DisableCopyOnRead^Read_20/ReadVariableOp^Read_21/DisableCopyOnRead^Read_21/ReadVariableOp^Read_22/DisableCopyOnRead^Read_22/ReadVariableOp^Read_23/DisableCopyOnRead^Read_23/ReadVariableOp^Read_24/DisableCopyOnRead^Read_24/ReadVariableOp^Read_25/DisableCopyOnRead^Read_25/ReadVariableOp^Read_26/DisableCopyOnRead^Read_26/ReadVariableOp^Read_27/DisableCopyOnRead^Read_27/ReadVariableOp^Read_28/DisableCopyOnRead^Read_28/ReadVariableOp^Read_29/DisableCopyOnRead^Read_29/ReadVariableOp^Read_3/DisableCopyOnRead^Read_3/ReadVariableOp^Read_30/DisableCopyOnRead^Read_30/ReadVariableOp^Read_31/DisableCopyOnRead^Read_31/ReadVariableOp^Read_32/DisableCopyOnRead^Read_32/ReadVariableOp^Read_33/DisableCopyOnRead^Read_33/ReadVariableOp^Read_34/DisableCopyOnRead^Read_34/ReadVariableOp^Read_35/DisableCopyOnRead^Read_35/ReadVariableOp^Read_36/DisableCopyOnRead^Read_36/ReadVariableOp^Read_37/DisableCopyOnRead^Read_37/ReadVariableOp^Read_38/DisableCopyOnRead^Read_38/ReadVariableOp^Read_39/DisableCopyOnRead^Read_39/ReadVariableOp^Read_4/DisableCopyOnRead^Read_4/ReadVariableOp^Read_40/DisableCopyOnRead^Read_40/ReadVariableOp^Read_41/DisableCopyOnRead^Read_41/ReadVariableOp^Read_42/DisableCopyOnRead^Read_42/ReadVariableOp^Read_43/DisableCopyOnRead^Read_43/ReadVariableOp^Read_44/DisableCopyOnRead^Read_44/ReadVariableOp^Read_45/DisableCopyOnRead^Read_45/ReadVariableOp^Read_46/DisableCopyOnRead^Read_46/ReadVariableOp^Read_47/DisableCopyOnRead^Read_47/ReadVariableOp^Read_48/DisableCopyOnRead^Read_48/ReadVariableOp^Read_49/DisableCopyOnRead^Read_49/ReadVariableOp^Read_5/DisableCopyOnRead^Read_5/ReadVariableOp^Read_50/DisableCopyOnRead^Read_50/ReadVariableOp^Read_51/DisableCopyOnRead^Read_51/ReadVariableOp^Read_52/DisableCopyOnRead^Read_52/ReadVariableOp^Read_53/DisableCopyOnRead^Read_53/ReadVariableOp^Read_54/DisableCopyOnRead^Read_54/ReadVariableOp^Read_55/DisableCopyOnRead^Read_55/ReadVariableOp^Read_6/DisableCopyOnRead^Read_6/ReadVariableOp^Read_7/DisableCopyOnRead^Read_7/ReadVariableOp^Read_8/DisableCopyOnRead^Read_8/ReadVariableOp^Read_9/DisableCopyOnRead^Read_9/ReadVariableOp*
_output_shapes
 "%
identity_113Identity_113:output:0*(
_construction_contextkEagerRuntime*�
_input_shapesv
t: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2(
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
Read_55/ReadVariableOpRead_55/ReadVariableOp24
Read_6/DisableCopyOnReadRead_6/DisableCopyOnRead2.
Read_6/ReadVariableOpRead_6/ReadVariableOp24
Read_7/DisableCopyOnReadRead_7/DisableCopyOnRead2.
Read_7/ReadVariableOpRead_7/ReadVariableOp24
Read_8/DisableCopyOnReadRead_8/DisableCopyOnRead2.
Read_8/ReadVariableOpRead_8/ReadVariableOp24
Read_9/DisableCopyOnReadRead_9/DisableCopyOnRead2.
Read_9/ReadVariableOpRead_9/ReadVariableOp:=99

_output_shapes
: 

_user_specified_nameConst:X8T
R
_user_specified_name:8transformer_encoder_29/Encoder-FeedForwardLayer_2_3/bias:Z7V
T
_user_specified_name<:transformer_encoder_29/Encoder-FeedForwardLayer_2_3/kernel:X6T
R
_user_specified_name:8transformer_encoder_29/Encoder-FeedForwardLayer_1_3/bias:Z5V
T
_user_specified_name<:transformer_encoder_29/Encoder-FeedForwardLayer_1_3/kernel:\4X
V
_user_specified_name><transformer_encoder_29/Encoder-2nd-NormalizationLayer-3/beta:]3Y
W
_user_specified_name?=transformer_encoder_29/Encoder-2nd-NormalizationLayer-3/gamma:\2X
V
_user_specified_name><transformer_encoder_29/Encoder-1st-NormalizationLayer-3/beta:]1Y
W
_user_specified_name?=transformer_encoder_29/Encoder-1st-NormalizationLayer-3/gamma:i0e
c
_user_specified_nameKItransformer_encoder_29/Encoder-SelfAttentionLayer-3/attention_output/bias:k/g
e
_user_specified_nameMKtransformer_encoder_29/Encoder-SelfAttentionLayer-3/attention_output/kernel:^.Z
X
_user_specified_name@>transformer_encoder_29/Encoder-SelfAttentionLayer-3/value/bias:`-\
Z
_user_specified_nameB@transformer_encoder_29/Encoder-SelfAttentionLayer-3/value/kernel:\,X
V
_user_specified_name><transformer_encoder_29/Encoder-SelfAttentionLayer-3/key/bias:^+Z
X
_user_specified_name@>transformer_encoder_29/Encoder-SelfAttentionLayer-3/key/kernel:^*Z
X
_user_specified_name@>transformer_encoder_29/Encoder-SelfAttentionLayer-3/query/bias:`)\
Z
_user_specified_nameB@transformer_encoder_29/Encoder-SelfAttentionLayer-3/query/kernel:X(T
R
_user_specified_name:8transformer_encoder_28/Encoder-FeedForwardLayer_2_2/bias:Z'V
T
_user_specified_name<:transformer_encoder_28/Encoder-FeedForwardLayer_2_2/kernel:X&T
R
_user_specified_name:8transformer_encoder_28/Encoder-FeedForwardLayer_1_2/bias:Z%V
T
_user_specified_name<:transformer_encoder_28/Encoder-FeedForwardLayer_1_2/kernel:\$X
V
_user_specified_name><transformer_encoder_28/Encoder-2nd-NormalizationLayer-2/beta:]#Y
W
_user_specified_name?=transformer_encoder_28/Encoder-2nd-NormalizationLayer-2/gamma:\"X
V
_user_specified_name><transformer_encoder_28/Encoder-1st-NormalizationLayer-2/beta:]!Y
W
_user_specified_name?=transformer_encoder_28/Encoder-1st-NormalizationLayer-2/gamma:i e
c
_user_specified_nameKItransformer_encoder_28/Encoder-SelfAttentionLayer-2/attention_output/bias:kg
e
_user_specified_nameMKtransformer_encoder_28/Encoder-SelfAttentionLayer-2/attention_output/kernel:^Z
X
_user_specified_name@>transformer_encoder_28/Encoder-SelfAttentionLayer-2/value/bias:`\
Z
_user_specified_nameB@transformer_encoder_28/Encoder-SelfAttentionLayer-2/value/kernel:\X
V
_user_specified_name><transformer_encoder_28/Encoder-SelfAttentionLayer-2/key/bias:^Z
X
_user_specified_name@>transformer_encoder_28/Encoder-SelfAttentionLayer-2/key/kernel:^Z
X
_user_specified_name@>transformer_encoder_28/Encoder-SelfAttentionLayer-2/query/bias:`\
Z
_user_specified_nameB@transformer_encoder_28/Encoder-SelfAttentionLayer-2/query/kernel:XT
R
_user_specified_name:8transformer_encoder_27/Encoder-FeedForwardLayer_2_1/bias:ZV
T
_user_specified_name<:transformer_encoder_27/Encoder-FeedForwardLayer_2_1/kernel:XT
R
_user_specified_name:8transformer_encoder_27/Encoder-FeedForwardLayer_1_1/bias:ZV
T
_user_specified_name<:transformer_encoder_27/Encoder-FeedForwardLayer_1_1/kernel:\X
V
_user_specified_name><transformer_encoder_27/Encoder-2nd-NormalizationLayer-1/beta:]Y
W
_user_specified_name?=transformer_encoder_27/Encoder-2nd-NormalizationLayer-1/gamma:\X
V
_user_specified_name><transformer_encoder_27/Encoder-1st-NormalizationLayer-1/beta:]Y
W
_user_specified_name?=transformer_encoder_27/Encoder-1st-NormalizationLayer-1/gamma:ie
c
_user_specified_nameKItransformer_encoder_27/Encoder-SelfAttentionLayer-1/attention_output/bias:kg
e
_user_specified_nameMKtransformer_encoder_27/Encoder-SelfAttentionLayer-1/attention_output/kernel:^Z
X
_user_specified_name@>transformer_encoder_27/Encoder-SelfAttentionLayer-1/value/bias:`\
Z
_user_specified_nameB@transformer_encoder_27/Encoder-SelfAttentionLayer-1/value/kernel:\X
V
_user_specified_name><transformer_encoder_27/Encoder-SelfAttentionLayer-1/key/bias:^Z
X
_user_specified_name@>transformer_encoder_27/Encoder-SelfAttentionLayer-1/key/kernel:^
Z
X
_user_specified_name@>transformer_encoder_27/Encoder-SelfAttentionLayer-1/query/bias:`	\
Z
_user_specified_nameB@transformer_encoder_27/Encoder-SelfAttentionLayer-1/query/kernel:84
2
_user_specified_namePredictionAreaRatio/bias::6
4
_user_specified_namePredictionAreaRatio/kernel:51
/
_user_specified_namePredictionStacks/bias:73
1
_user_specified_namePredictionStacks/kernel:A=
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
�
^
B__inference_ReduceStackDimensionViaSummation_layer_call_fn_2631785

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
]__inference_ReduceStackDimensionViaSummation_layer_call_and_return_conditional_losses_2629664`
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
D
(__inference_Output_layer_call_fn_2631947

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
C__inference_Output_layer_call_and_return_conditional_losses_2629702Q
IdentityIdentityPartitionedCall:output:0*
T0*
_output_shapes
:"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������P:S O
+
_output_shapes
:���������P
 
_user_specified_nameinputs
�
R
6__inference_StandardizeTimeLimit_layer_call_fn_2631811

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
Q__inference_StandardizeTimeLimit_layer_call_and_return_conditional_losses_2629674`
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
�
�
5__inference_PredictionAreaRatio_layer_call_fn_2631916

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
P__inference_PredictionAreaRatio_layer_call_and_return_conditional_losses_2628966o
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
_user_specified_name	2631912:'#
!
_user_specified_name	2631910:O K
'
_output_shapes
:���������

 
_user_specified_nameinputs
��
�
S__inference_transformer_encoder_27_layer_call_and_return_conditional_losses_2630844

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
dropout_27/IdentityIdentity-Encoder-FeedForwardLayer_2_1/BiasAdd:output:0*
T0*+
_output_shapes
:���������P	�
Encoder-2nd-AdditionLayer-1/addAddV2#Encoder-1st-AdditionLayer-1/add:z:0dropout_27/Identity:output:0*
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
�
_
C__inference_Output_layer_call_and_return_conditional_losses_2631957

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
_construction_contextkEagerRuntime**
_input_shapes
:���������P:S O
+
_output_shapes
:���������P
 
_user_specified_nameinputs
�
m
Q__inference_StandardizeTimeLimit_layer_call_and_return_conditional_losses_2629674

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
��	
�U
"__inference__wrapped_model_2628176
stacklevelinputfeatures
timelimitinputi
[model_9_transformer_encoder_27_encoder_1st_normalizationlayer_1_mul_readvariableop_resource:	i
[model_9_transformer_encoder_27_encoder_1st_normalizationlayer_1_add_readvariableop_resource:	}
gmodel_9_transformer_encoder_27_encoder_selfattentionlayer_1_query_einsum_einsum_readvariableop_resource:	o
]model_9_transformer_encoder_27_encoder_selfattentionlayer_1_query_add_readvariableop_resource:{
emodel_9_transformer_encoder_27_encoder_selfattentionlayer_1_key_einsum_einsum_readvariableop_resource:	m
[model_9_transformer_encoder_27_encoder_selfattentionlayer_1_key_add_readvariableop_resource:}
gmodel_9_transformer_encoder_27_encoder_selfattentionlayer_1_value_einsum_einsum_readvariableop_resource:	o
]model_9_transformer_encoder_27_encoder_selfattentionlayer_1_value_add_readvariableop_resource:�
rmodel_9_transformer_encoder_27_encoder_selfattentionlayer_1_attention_output_einsum_einsum_readvariableop_resource:	v
hmodel_9_transformer_encoder_27_encoder_selfattentionlayer_1_attention_output_add_readvariableop_resource:	i
[model_9_transformer_encoder_27_encoder_2nd_normalizationlayer_1_mul_readvariableop_resource:	i
[model_9_transformer_encoder_27_encoder_2nd_normalizationlayer_1_add_readvariableop_resource:	o
]model_9_transformer_encoder_27_encoder_feedforwardlayer_1_1_tensordot_readvariableop_resource:		i
[model_9_transformer_encoder_27_encoder_feedforwardlayer_1_1_biasadd_readvariableop_resource:	o
]model_9_transformer_encoder_27_encoder_feedforwardlayer_2_1_tensordot_readvariableop_resource:		i
[model_9_transformer_encoder_27_encoder_feedforwardlayer_2_1_biasadd_readvariableop_resource:	i
[model_9_transformer_encoder_28_encoder_1st_normalizationlayer_2_mul_readvariableop_resource:	i
[model_9_transformer_encoder_28_encoder_1st_normalizationlayer_2_add_readvariableop_resource:	}
gmodel_9_transformer_encoder_28_encoder_selfattentionlayer_2_query_einsum_einsum_readvariableop_resource:	o
]model_9_transformer_encoder_28_encoder_selfattentionlayer_2_query_add_readvariableop_resource:{
emodel_9_transformer_encoder_28_encoder_selfattentionlayer_2_key_einsum_einsum_readvariableop_resource:	m
[model_9_transformer_encoder_28_encoder_selfattentionlayer_2_key_add_readvariableop_resource:}
gmodel_9_transformer_encoder_28_encoder_selfattentionlayer_2_value_einsum_einsum_readvariableop_resource:	o
]model_9_transformer_encoder_28_encoder_selfattentionlayer_2_value_add_readvariableop_resource:�
rmodel_9_transformer_encoder_28_encoder_selfattentionlayer_2_attention_output_einsum_einsum_readvariableop_resource:	v
hmodel_9_transformer_encoder_28_encoder_selfattentionlayer_2_attention_output_add_readvariableop_resource:	i
[model_9_transformer_encoder_28_encoder_2nd_normalizationlayer_2_mul_readvariableop_resource:	i
[model_9_transformer_encoder_28_encoder_2nd_normalizationlayer_2_add_readvariableop_resource:	o
]model_9_transformer_encoder_28_encoder_feedforwardlayer_1_2_tensordot_readvariableop_resource:		i
[model_9_transformer_encoder_28_encoder_feedforwardlayer_1_2_biasadd_readvariableop_resource:	o
]model_9_transformer_encoder_28_encoder_feedforwardlayer_2_2_tensordot_readvariableop_resource:		i
[model_9_transformer_encoder_28_encoder_feedforwardlayer_2_2_biasadd_readvariableop_resource:	i
[model_9_transformer_encoder_29_encoder_1st_normalizationlayer_3_mul_readvariableop_resource:	i
[model_9_transformer_encoder_29_encoder_1st_normalizationlayer_3_add_readvariableop_resource:	}
gmodel_9_transformer_encoder_29_encoder_selfattentionlayer_3_query_einsum_einsum_readvariableop_resource:	o
]model_9_transformer_encoder_29_encoder_selfattentionlayer_3_query_add_readvariableop_resource:{
emodel_9_transformer_encoder_29_encoder_selfattentionlayer_3_key_einsum_einsum_readvariableop_resource:	m
[model_9_transformer_encoder_29_encoder_selfattentionlayer_3_key_add_readvariableop_resource:}
gmodel_9_transformer_encoder_29_encoder_selfattentionlayer_3_value_einsum_einsum_readvariableop_resource:	o
]model_9_transformer_encoder_29_encoder_selfattentionlayer_3_value_add_readvariableop_resource:�
rmodel_9_transformer_encoder_29_encoder_selfattentionlayer_3_attention_output_einsum_einsum_readvariableop_resource:	v
hmodel_9_transformer_encoder_29_encoder_selfattentionlayer_3_attention_output_add_readvariableop_resource:	i
[model_9_transformer_encoder_29_encoder_2nd_normalizationlayer_3_mul_readvariableop_resource:	i
[model_9_transformer_encoder_29_encoder_2nd_normalizationlayer_3_add_readvariableop_resource:	o
]model_9_transformer_encoder_29_encoder_feedforwardlayer_1_3_tensordot_readvariableop_resource:		i
[model_9_transformer_encoder_29_encoder_feedforwardlayer_1_3_biasadd_readvariableop_resource:	o
]model_9_transformer_encoder_29_encoder_feedforwardlayer_2_3_tensordot_readvariableop_resource:		i
[model_9_transformer_encoder_29_encoder_feedforwardlayer_2_3_biasadd_readvariableop_resource:	@
2model_9_finallayernorm_mul_readvariableop_resource:	@
2model_9_finallayernorm_add_readvariableop_resource:	U
Cmodel_9_fullyconnectedlayerarearatio_matmul_readvariableop_resource:

R
Dmodel_9_fullyconnectedlayerarearatio_biasadd_readvariableop_resource:
L
:model_9_predictionarearatio_matmul_readvariableop_resource:
I
;model_9_predictionarearatio_biasadd_readvariableop_resource:L
:model_9_predictionstacks_tensordot_readvariableop_resource:	F
8model_9_predictionstacks_biasadd_readvariableop_resource:
identity

identity_1��)model_9/FinalLayerNorm/add/ReadVariableOp�)model_9/FinalLayerNorm/mul/ReadVariableOp�;model_9/FullyConnectedLayerAreaRatio/BiasAdd/ReadVariableOp�:model_9/FullyConnectedLayerAreaRatio/MatMul/ReadVariableOp�2model_9/PredictionAreaRatio/BiasAdd/ReadVariableOp�1model_9/PredictionAreaRatio/MatMul/ReadVariableOp�/model_9/PredictionStacks/BiasAdd/ReadVariableOp�1model_9/PredictionStacks/Tensordot/ReadVariableOp�Rmodel_9/transformer_encoder_27/Encoder-1st-NormalizationLayer-1/add/ReadVariableOp�Rmodel_9/transformer_encoder_27/Encoder-1st-NormalizationLayer-1/mul/ReadVariableOp�Rmodel_9/transformer_encoder_27/Encoder-2nd-NormalizationLayer-1/add/ReadVariableOp�Rmodel_9/transformer_encoder_27/Encoder-2nd-NormalizationLayer-1/mul/ReadVariableOp�Rmodel_9/transformer_encoder_27/Encoder-FeedForwardLayer_1_1/BiasAdd/ReadVariableOp�Tmodel_9/transformer_encoder_27/Encoder-FeedForwardLayer_1_1/Tensordot/ReadVariableOp�Rmodel_9/transformer_encoder_27/Encoder-FeedForwardLayer_2_1/BiasAdd/ReadVariableOp�Tmodel_9/transformer_encoder_27/Encoder-FeedForwardLayer_2_1/Tensordot/ReadVariableOp�_model_9/transformer_encoder_27/Encoder-SelfAttentionLayer-1/attention_output/add/ReadVariableOp�imodel_9/transformer_encoder_27/Encoder-SelfAttentionLayer-1/attention_output/einsum/Einsum/ReadVariableOp�Rmodel_9/transformer_encoder_27/Encoder-SelfAttentionLayer-1/key/add/ReadVariableOp�\model_9/transformer_encoder_27/Encoder-SelfAttentionLayer-1/key/einsum/Einsum/ReadVariableOp�Tmodel_9/transformer_encoder_27/Encoder-SelfAttentionLayer-1/query/add/ReadVariableOp�^model_9/transformer_encoder_27/Encoder-SelfAttentionLayer-1/query/einsum/Einsum/ReadVariableOp�Tmodel_9/transformer_encoder_27/Encoder-SelfAttentionLayer-1/value/add/ReadVariableOp�^model_9/transformer_encoder_27/Encoder-SelfAttentionLayer-1/value/einsum/Einsum/ReadVariableOp�Rmodel_9/transformer_encoder_28/Encoder-1st-NormalizationLayer-2/add/ReadVariableOp�Rmodel_9/transformer_encoder_28/Encoder-1st-NormalizationLayer-2/mul/ReadVariableOp�Rmodel_9/transformer_encoder_28/Encoder-2nd-NormalizationLayer-2/add/ReadVariableOp�Rmodel_9/transformer_encoder_28/Encoder-2nd-NormalizationLayer-2/mul/ReadVariableOp�Rmodel_9/transformer_encoder_28/Encoder-FeedForwardLayer_1_2/BiasAdd/ReadVariableOp�Tmodel_9/transformer_encoder_28/Encoder-FeedForwardLayer_1_2/Tensordot/ReadVariableOp�Rmodel_9/transformer_encoder_28/Encoder-FeedForwardLayer_2_2/BiasAdd/ReadVariableOp�Tmodel_9/transformer_encoder_28/Encoder-FeedForwardLayer_2_2/Tensordot/ReadVariableOp�_model_9/transformer_encoder_28/Encoder-SelfAttentionLayer-2/attention_output/add/ReadVariableOp�imodel_9/transformer_encoder_28/Encoder-SelfAttentionLayer-2/attention_output/einsum/Einsum/ReadVariableOp�Rmodel_9/transformer_encoder_28/Encoder-SelfAttentionLayer-2/key/add/ReadVariableOp�\model_9/transformer_encoder_28/Encoder-SelfAttentionLayer-2/key/einsum/Einsum/ReadVariableOp�Tmodel_9/transformer_encoder_28/Encoder-SelfAttentionLayer-2/query/add/ReadVariableOp�^model_9/transformer_encoder_28/Encoder-SelfAttentionLayer-2/query/einsum/Einsum/ReadVariableOp�Tmodel_9/transformer_encoder_28/Encoder-SelfAttentionLayer-2/value/add/ReadVariableOp�^model_9/transformer_encoder_28/Encoder-SelfAttentionLayer-2/value/einsum/Einsum/ReadVariableOp�Rmodel_9/transformer_encoder_29/Encoder-1st-NormalizationLayer-3/add/ReadVariableOp�Rmodel_9/transformer_encoder_29/Encoder-1st-NormalizationLayer-3/mul/ReadVariableOp�Rmodel_9/transformer_encoder_29/Encoder-2nd-NormalizationLayer-3/add/ReadVariableOp�Rmodel_9/transformer_encoder_29/Encoder-2nd-NormalizationLayer-3/mul/ReadVariableOp�Rmodel_9/transformer_encoder_29/Encoder-FeedForwardLayer_1_3/BiasAdd/ReadVariableOp�Tmodel_9/transformer_encoder_29/Encoder-FeedForwardLayer_1_3/Tensordot/ReadVariableOp�Rmodel_9/transformer_encoder_29/Encoder-FeedForwardLayer_2_3/BiasAdd/ReadVariableOp�Tmodel_9/transformer_encoder_29/Encoder-FeedForwardLayer_2_3/Tensordot/ReadVariableOp�_model_9/transformer_encoder_29/Encoder-SelfAttentionLayer-3/attention_output/add/ReadVariableOp�imodel_9/transformer_encoder_29/Encoder-SelfAttentionLayer-3/attention_output/einsum/Einsum/ReadVariableOp�Rmodel_9/transformer_encoder_29/Encoder-SelfAttentionLayer-3/key/add/ReadVariableOp�\model_9/transformer_encoder_29/Encoder-SelfAttentionLayer-3/key/einsum/Einsum/ReadVariableOp�Tmodel_9/transformer_encoder_29/Encoder-SelfAttentionLayer-3/query/add/ReadVariableOp�^model_9/transformer_encoder_29/Encoder-SelfAttentionLayer-3/query/einsum/Einsum/ReadVariableOp�Tmodel_9/transformer_encoder_29/Encoder-SelfAttentionLayer-3/value/add/ReadVariableOp�^model_9/transformer_encoder_29/Encoder-SelfAttentionLayer-3/value/einsum/Einsum/ReadVariableOpa
model_9/MaskingLayer/NotEqual/yConst*
_output_shapes
: *
dtype0*
value	B j �
model_9/MaskingLayer/NotEqualNotEqualstacklevelinputfeatures(model_9/MaskingLayer/NotEqual/y:output:0*
T0*+
_output_shapes
:���������P	u
*model_9/MaskingLayer/Any/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
����������
model_9/MaskingLayer/AnyAny!model_9/MaskingLayer/NotEqual:z:03model_9/MaskingLayer/Any/reduction_indices:output:0*+
_output_shapes
:���������P*
	keep_dims(�
model_9/MaskingLayer/CastCast!model_9/MaskingLayer/Any:output:0*

DstT0*

SrcT0
*+
_output_shapes
:���������P�
model_9/MaskingLayer/mulMulstacklevelinputfeaturesmodel_9/MaskingLayer/Cast:y:0*
T0*+
_output_shapes
:���������P	�
model_9/MaskingLayer/SqueezeSqueeze!model_9/MaskingLayer/Any:output:0*
T0
*'
_output_shapes
:���������P*
squeeze_dims

����������
#model_9/transformer_encoder_27/CastCastmodel_9/MaskingLayer/mul:z:0*

DstT0*

SrcT0*+
_output_shapes
:���������P	�
Emodel_9/transformer_encoder_27/Encoder-1st-NormalizationLayer-1/ShapeShape'model_9/transformer_encoder_27/Cast:y:0*
T0*
_output_shapes
::���
Smodel_9/transformer_encoder_27/Encoder-1st-NormalizationLayer-1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
Umodel_9/transformer_encoder_27/Encoder-1st-NormalizationLayer-1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
Umodel_9/transformer_encoder_27/Encoder-1st-NormalizationLayer-1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
Mmodel_9/transformer_encoder_27/Encoder-1st-NormalizationLayer-1/strided_sliceStridedSliceNmodel_9/transformer_encoder_27/Encoder-1st-NormalizationLayer-1/Shape:output:0\model_9/transformer_encoder_27/Encoder-1st-NormalizationLayer-1/strided_slice/stack:output:0^model_9/transformer_encoder_27/Encoder-1st-NormalizationLayer-1/strided_slice/stack_1:output:0^model_9/transformer_encoder_27/Encoder-1st-NormalizationLayer-1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask�
Emodel_9/transformer_encoder_27/Encoder-1st-NormalizationLayer-1/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
Dmodel_9/transformer_encoder_27/Encoder-1st-NormalizationLayer-1/ProdProdVmodel_9/transformer_encoder_27/Encoder-1st-NormalizationLayer-1/strided_slice:output:0Nmodel_9/transformer_encoder_27/Encoder-1st-NormalizationLayer-1/Const:output:0*
T0*
_output_shapes
: �
Umodel_9/transformer_encoder_27/Encoder-1st-NormalizationLayer-1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:�
Wmodel_9/transformer_encoder_27/Encoder-1st-NormalizationLayer-1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: �
Wmodel_9/transformer_encoder_27/Encoder-1st-NormalizationLayer-1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
Omodel_9/transformer_encoder_27/Encoder-1st-NormalizationLayer-1/strided_slice_1StridedSliceNmodel_9/transformer_encoder_27/Encoder-1st-NormalizationLayer-1/Shape:output:0^model_9/transformer_encoder_27/Encoder-1st-NormalizationLayer-1/strided_slice_1/stack:output:0`model_9/transformer_encoder_27/Encoder-1st-NormalizationLayer-1/strided_slice_1/stack_1:output:0`model_9/transformer_encoder_27/Encoder-1st-NormalizationLayer-1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask�
Gmodel_9/transformer_encoder_27/Encoder-1st-NormalizationLayer-1/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
Fmodel_9/transformer_encoder_27/Encoder-1st-NormalizationLayer-1/Prod_1ProdXmodel_9/transformer_encoder_27/Encoder-1st-NormalizationLayer-1/strided_slice_1:output:0Pmodel_9/transformer_encoder_27/Encoder-1st-NormalizationLayer-1/Const_1:output:0*
T0*
_output_shapes
: �
Omodel_9/transformer_encoder_27/Encoder-1st-NormalizationLayer-1/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :�
Omodel_9/transformer_encoder_27/Encoder-1st-NormalizationLayer-1/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :�
Mmodel_9/transformer_encoder_27/Encoder-1st-NormalizationLayer-1/Reshape/shapePackXmodel_9/transformer_encoder_27/Encoder-1st-NormalizationLayer-1/Reshape/shape/0:output:0Mmodel_9/transformer_encoder_27/Encoder-1st-NormalizationLayer-1/Prod:output:0Omodel_9/transformer_encoder_27/Encoder-1st-NormalizationLayer-1/Prod_1:output:0Xmodel_9/transformer_encoder_27/Encoder-1st-NormalizationLayer-1/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:�
Gmodel_9/transformer_encoder_27/Encoder-1st-NormalizationLayer-1/ReshapeReshape'model_9/transformer_encoder_27/Cast:y:0Vmodel_9/transformer_encoder_27/Encoder-1st-NormalizationLayer-1/Reshape/shape:output:0*
T0*8
_output_shapes&
$:"�������������������
Kmodel_9/transformer_encoder_27/Encoder-1st-NormalizationLayer-1/ones/packedPackMmodel_9/transformer_encoder_27/Encoder-1st-NormalizationLayer-1/Prod:output:0*
N*
T0*
_output_shapes
:�
Jmodel_9/transformer_encoder_27/Encoder-1st-NormalizationLayer-1/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
Dmodel_9/transformer_encoder_27/Encoder-1st-NormalizationLayer-1/onesFillTmodel_9/transformer_encoder_27/Encoder-1st-NormalizationLayer-1/ones/packed:output:0Smodel_9/transformer_encoder_27/Encoder-1st-NormalizationLayer-1/ones/Const:output:0*
T0*#
_output_shapes
:����������
Lmodel_9/transformer_encoder_27/Encoder-1st-NormalizationLayer-1/zeros/packedPackMmodel_9/transformer_encoder_27/Encoder-1st-NormalizationLayer-1/Prod:output:0*
N*
T0*
_output_shapes
:�
Kmodel_9/transformer_encoder_27/Encoder-1st-NormalizationLayer-1/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
Emodel_9/transformer_encoder_27/Encoder-1st-NormalizationLayer-1/zerosFillUmodel_9/transformer_encoder_27/Encoder-1st-NormalizationLayer-1/zeros/packed:output:0Tmodel_9/transformer_encoder_27/Encoder-1st-NormalizationLayer-1/zeros/Const:output:0*
T0*#
_output_shapes
:����������
Gmodel_9/transformer_encoder_27/Encoder-1st-NormalizationLayer-1/Const_2Const*
_output_shapes
: *
dtype0*
valueB �
Gmodel_9/transformer_encoder_27/Encoder-1st-NormalizationLayer-1/Const_3Const*
_output_shapes
: *
dtype0*
valueB �
Pmodel_9/transformer_encoder_27/Encoder-1st-NormalizationLayer-1/FusedBatchNormV3FusedBatchNormV3Pmodel_9/transformer_encoder_27/Encoder-1st-NormalizationLayer-1/Reshape:output:0Mmodel_9/transformer_encoder_27/Encoder-1st-NormalizationLayer-1/ones:output:0Nmodel_9/transformer_encoder_27/Encoder-1st-NormalizationLayer-1/zeros:output:0Pmodel_9/transformer_encoder_27/Encoder-1st-NormalizationLayer-1/Const_2:output:0Pmodel_9/transformer_encoder_27/Encoder-1st-NormalizationLayer-1/Const_3:output:0*
T0*
U0*x
_output_shapesf
d:"������������������:���������:���������:���������:���������:*
data_formatNCHW*
epsilon%o�:�
Imodel_9/transformer_encoder_27/Encoder-1st-NormalizationLayer-1/Reshape_1ReshapeTmodel_9/transformer_encoder_27/Encoder-1st-NormalizationLayer-1/FusedBatchNormV3:y:0Nmodel_9/transformer_encoder_27/Encoder-1st-NormalizationLayer-1/Shape:output:0*
T0*+
_output_shapes
:���������P	�
Rmodel_9/transformer_encoder_27/Encoder-1st-NormalizationLayer-1/mul/ReadVariableOpReadVariableOp[model_9_transformer_encoder_27_encoder_1st_normalizationlayer_1_mul_readvariableop_resource*
_output_shapes
:	*
dtype0�
Cmodel_9/transformer_encoder_27/Encoder-1st-NormalizationLayer-1/mulMulRmodel_9/transformer_encoder_27/Encoder-1st-NormalizationLayer-1/Reshape_1:output:0Zmodel_9/transformer_encoder_27/Encoder-1st-NormalizationLayer-1/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������P	�
Rmodel_9/transformer_encoder_27/Encoder-1st-NormalizationLayer-1/add/ReadVariableOpReadVariableOp[model_9_transformer_encoder_27_encoder_1st_normalizationlayer_1_add_readvariableop_resource*
_output_shapes
:	*
dtype0�
Cmodel_9/transformer_encoder_27/Encoder-1st-NormalizationLayer-1/addAddV2Gmodel_9/transformer_encoder_27/Encoder-1st-NormalizationLayer-1/mul:z:0Zmodel_9/transformer_encoder_27/Encoder-1st-NormalizationLayer-1/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������P	�
^model_9/transformer_encoder_27/Encoder-SelfAttentionLayer-1/query/einsum/Einsum/ReadVariableOpReadVariableOpgmodel_9_transformer_encoder_27_encoder_selfattentionlayer_1_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:	*
dtype0�
Omodel_9/transformer_encoder_27/Encoder-SelfAttentionLayer-1/query/einsum/EinsumEinsumGmodel_9/transformer_encoder_27/Encoder-1st-NormalizationLayer-1/add:z:0fmodel_9/transformer_encoder_27/Encoder-SelfAttentionLayer-1/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������P*
equationabc,cde->abde�
Tmodel_9/transformer_encoder_27/Encoder-SelfAttentionLayer-1/query/add/ReadVariableOpReadVariableOp]model_9_transformer_encoder_27_encoder_selfattentionlayer_1_query_add_readvariableop_resource*
_output_shapes

:*
dtype0�
Emodel_9/transformer_encoder_27/Encoder-SelfAttentionLayer-1/query/addAddV2Xmodel_9/transformer_encoder_27/Encoder-SelfAttentionLayer-1/query/einsum/Einsum:output:0\model_9/transformer_encoder_27/Encoder-SelfAttentionLayer-1/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������P�
\model_9/transformer_encoder_27/Encoder-SelfAttentionLayer-1/key/einsum/Einsum/ReadVariableOpReadVariableOpemodel_9_transformer_encoder_27_encoder_selfattentionlayer_1_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:	*
dtype0�
Mmodel_9/transformer_encoder_27/Encoder-SelfAttentionLayer-1/key/einsum/EinsumEinsumGmodel_9/transformer_encoder_27/Encoder-1st-NormalizationLayer-1/add:z:0dmodel_9/transformer_encoder_27/Encoder-SelfAttentionLayer-1/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������P*
equationabc,cde->abde�
Rmodel_9/transformer_encoder_27/Encoder-SelfAttentionLayer-1/key/add/ReadVariableOpReadVariableOp[model_9_transformer_encoder_27_encoder_selfattentionlayer_1_key_add_readvariableop_resource*
_output_shapes

:*
dtype0�
Cmodel_9/transformer_encoder_27/Encoder-SelfAttentionLayer-1/key/addAddV2Vmodel_9/transformer_encoder_27/Encoder-SelfAttentionLayer-1/key/einsum/Einsum:output:0Zmodel_9/transformer_encoder_27/Encoder-SelfAttentionLayer-1/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������P�
^model_9/transformer_encoder_27/Encoder-SelfAttentionLayer-1/value/einsum/Einsum/ReadVariableOpReadVariableOpgmodel_9_transformer_encoder_27_encoder_selfattentionlayer_1_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:	*
dtype0�
Omodel_9/transformer_encoder_27/Encoder-SelfAttentionLayer-1/value/einsum/EinsumEinsumGmodel_9/transformer_encoder_27/Encoder-1st-NormalizationLayer-1/add:z:0fmodel_9/transformer_encoder_27/Encoder-SelfAttentionLayer-1/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������P*
equationabc,cde->abde�
Tmodel_9/transformer_encoder_27/Encoder-SelfAttentionLayer-1/value/add/ReadVariableOpReadVariableOp]model_9_transformer_encoder_27_encoder_selfattentionlayer_1_value_add_readvariableop_resource*
_output_shapes

:*
dtype0�
Emodel_9/transformer_encoder_27/Encoder-SelfAttentionLayer-1/value/addAddV2Xmodel_9/transformer_encoder_27/Encoder-SelfAttentionLayer-1/value/einsum/Einsum:output:0\model_9/transformer_encoder_27/Encoder-SelfAttentionLayer-1/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������P�
Amodel_9/transformer_encoder_27/Encoder-SelfAttentionLayer-1/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *:�?�
?model_9/transformer_encoder_27/Encoder-SelfAttentionLayer-1/MulMulImodel_9/transformer_encoder_27/Encoder-SelfAttentionLayer-1/query/add:z:0Jmodel_9/transformer_encoder_27/Encoder-SelfAttentionLayer-1/Mul/y:output:0*
T0*/
_output_shapes
:���������P�
Imodel_9/transformer_encoder_27/Encoder-SelfAttentionLayer-1/einsum/EinsumEinsumGmodel_9/transformer_encoder_27/Encoder-SelfAttentionLayer-1/key/add:z:0Cmodel_9/transformer_encoder_27/Encoder-SelfAttentionLayer-1/Mul:z:0*
N*
T0*/
_output_shapes
:���������PP*
equationaecd,abcd->acbe�
Kmodel_9/transformer_encoder_27/Encoder-SelfAttentionLayer-1/softmax/SoftmaxSoftmaxRmodel_9/transformer_encoder_27/Encoder-SelfAttentionLayer-1/einsum/Einsum:output:0*
T0*/
_output_shapes
:���������PP�
Lmodel_9/transformer_encoder_27/Encoder-SelfAttentionLayer-1/dropout/IdentityIdentityUmodel_9/transformer_encoder_27/Encoder-SelfAttentionLayer-1/softmax/Softmax:softmax:0*
T0*/
_output_shapes
:���������PP�
Kmodel_9/transformer_encoder_27/Encoder-SelfAttentionLayer-1/einsum_1/EinsumEinsumUmodel_9/transformer_encoder_27/Encoder-SelfAttentionLayer-1/dropout/Identity:output:0Imodel_9/transformer_encoder_27/Encoder-SelfAttentionLayer-1/value/add:z:0*
N*
T0*/
_output_shapes
:���������P*
equationacbe,aecd->abcd�
imodel_9/transformer_encoder_27/Encoder-SelfAttentionLayer-1/attention_output/einsum/Einsum/ReadVariableOpReadVariableOprmodel_9_transformer_encoder_27_encoder_selfattentionlayer_1_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:	*
dtype0�
Zmodel_9/transformer_encoder_27/Encoder-SelfAttentionLayer-1/attention_output/einsum/EinsumEinsumTmodel_9/transformer_encoder_27/Encoder-SelfAttentionLayer-1/einsum_1/Einsum:output:0qmodel_9/transformer_encoder_27/Encoder-SelfAttentionLayer-1/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:���������P	*
equationabcd,cde->abe�
_model_9/transformer_encoder_27/Encoder-SelfAttentionLayer-1/attention_output/add/ReadVariableOpReadVariableOphmodel_9_transformer_encoder_27_encoder_selfattentionlayer_1_attention_output_add_readvariableop_resource*
_output_shapes
:	*
dtype0�
Pmodel_9/transformer_encoder_27/Encoder-SelfAttentionLayer-1/attention_output/addAddV2cmodel_9/transformer_encoder_27/Encoder-SelfAttentionLayer-1/attention_output/einsum/Einsum:output:0gmodel_9/transformer_encoder_27/Encoder-SelfAttentionLayer-1/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������P	�
>model_9/transformer_encoder_27/Encoder-1st-AdditionLayer-1/addAddV2'model_9/transformer_encoder_27/Cast:y:0Tmodel_9/transformer_encoder_27/Encoder-SelfAttentionLayer-1/attention_output/add:z:0*
T0*+
_output_shapes
:���������P	�
Emodel_9/transformer_encoder_27/Encoder-2nd-NormalizationLayer-1/ShapeShapeBmodel_9/transformer_encoder_27/Encoder-1st-AdditionLayer-1/add:z:0*
T0*
_output_shapes
::���
Smodel_9/transformer_encoder_27/Encoder-2nd-NormalizationLayer-1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
Umodel_9/transformer_encoder_27/Encoder-2nd-NormalizationLayer-1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
Umodel_9/transformer_encoder_27/Encoder-2nd-NormalizationLayer-1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
Mmodel_9/transformer_encoder_27/Encoder-2nd-NormalizationLayer-1/strided_sliceStridedSliceNmodel_9/transformer_encoder_27/Encoder-2nd-NormalizationLayer-1/Shape:output:0\model_9/transformer_encoder_27/Encoder-2nd-NormalizationLayer-1/strided_slice/stack:output:0^model_9/transformer_encoder_27/Encoder-2nd-NormalizationLayer-1/strided_slice/stack_1:output:0^model_9/transformer_encoder_27/Encoder-2nd-NormalizationLayer-1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask�
Emodel_9/transformer_encoder_27/Encoder-2nd-NormalizationLayer-1/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
Dmodel_9/transformer_encoder_27/Encoder-2nd-NormalizationLayer-1/ProdProdVmodel_9/transformer_encoder_27/Encoder-2nd-NormalizationLayer-1/strided_slice:output:0Nmodel_9/transformer_encoder_27/Encoder-2nd-NormalizationLayer-1/Const:output:0*
T0*
_output_shapes
: �
Umodel_9/transformer_encoder_27/Encoder-2nd-NormalizationLayer-1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:�
Wmodel_9/transformer_encoder_27/Encoder-2nd-NormalizationLayer-1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: �
Wmodel_9/transformer_encoder_27/Encoder-2nd-NormalizationLayer-1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
Omodel_9/transformer_encoder_27/Encoder-2nd-NormalizationLayer-1/strided_slice_1StridedSliceNmodel_9/transformer_encoder_27/Encoder-2nd-NormalizationLayer-1/Shape:output:0^model_9/transformer_encoder_27/Encoder-2nd-NormalizationLayer-1/strided_slice_1/stack:output:0`model_9/transformer_encoder_27/Encoder-2nd-NormalizationLayer-1/strided_slice_1/stack_1:output:0`model_9/transformer_encoder_27/Encoder-2nd-NormalizationLayer-1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask�
Gmodel_9/transformer_encoder_27/Encoder-2nd-NormalizationLayer-1/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
Fmodel_9/transformer_encoder_27/Encoder-2nd-NormalizationLayer-1/Prod_1ProdXmodel_9/transformer_encoder_27/Encoder-2nd-NormalizationLayer-1/strided_slice_1:output:0Pmodel_9/transformer_encoder_27/Encoder-2nd-NormalizationLayer-1/Const_1:output:0*
T0*
_output_shapes
: �
Omodel_9/transformer_encoder_27/Encoder-2nd-NormalizationLayer-1/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :�
Omodel_9/transformer_encoder_27/Encoder-2nd-NormalizationLayer-1/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :�
Mmodel_9/transformer_encoder_27/Encoder-2nd-NormalizationLayer-1/Reshape/shapePackXmodel_9/transformer_encoder_27/Encoder-2nd-NormalizationLayer-1/Reshape/shape/0:output:0Mmodel_9/transformer_encoder_27/Encoder-2nd-NormalizationLayer-1/Prod:output:0Omodel_9/transformer_encoder_27/Encoder-2nd-NormalizationLayer-1/Prod_1:output:0Xmodel_9/transformer_encoder_27/Encoder-2nd-NormalizationLayer-1/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:�
Gmodel_9/transformer_encoder_27/Encoder-2nd-NormalizationLayer-1/ReshapeReshapeBmodel_9/transformer_encoder_27/Encoder-1st-AdditionLayer-1/add:z:0Vmodel_9/transformer_encoder_27/Encoder-2nd-NormalizationLayer-1/Reshape/shape:output:0*
T0*8
_output_shapes&
$:"�������������������
Kmodel_9/transformer_encoder_27/Encoder-2nd-NormalizationLayer-1/ones/packedPackMmodel_9/transformer_encoder_27/Encoder-2nd-NormalizationLayer-1/Prod:output:0*
N*
T0*
_output_shapes
:�
Jmodel_9/transformer_encoder_27/Encoder-2nd-NormalizationLayer-1/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
Dmodel_9/transformer_encoder_27/Encoder-2nd-NormalizationLayer-1/onesFillTmodel_9/transformer_encoder_27/Encoder-2nd-NormalizationLayer-1/ones/packed:output:0Smodel_9/transformer_encoder_27/Encoder-2nd-NormalizationLayer-1/ones/Const:output:0*
T0*#
_output_shapes
:����������
Lmodel_9/transformer_encoder_27/Encoder-2nd-NormalizationLayer-1/zeros/packedPackMmodel_9/transformer_encoder_27/Encoder-2nd-NormalizationLayer-1/Prod:output:0*
N*
T0*
_output_shapes
:�
Kmodel_9/transformer_encoder_27/Encoder-2nd-NormalizationLayer-1/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
Emodel_9/transformer_encoder_27/Encoder-2nd-NormalizationLayer-1/zerosFillUmodel_9/transformer_encoder_27/Encoder-2nd-NormalizationLayer-1/zeros/packed:output:0Tmodel_9/transformer_encoder_27/Encoder-2nd-NormalizationLayer-1/zeros/Const:output:0*
T0*#
_output_shapes
:����������
Gmodel_9/transformer_encoder_27/Encoder-2nd-NormalizationLayer-1/Const_2Const*
_output_shapes
: *
dtype0*
valueB �
Gmodel_9/transformer_encoder_27/Encoder-2nd-NormalizationLayer-1/Const_3Const*
_output_shapes
: *
dtype0*
valueB �
Pmodel_9/transformer_encoder_27/Encoder-2nd-NormalizationLayer-1/FusedBatchNormV3FusedBatchNormV3Pmodel_9/transformer_encoder_27/Encoder-2nd-NormalizationLayer-1/Reshape:output:0Mmodel_9/transformer_encoder_27/Encoder-2nd-NormalizationLayer-1/ones:output:0Nmodel_9/transformer_encoder_27/Encoder-2nd-NormalizationLayer-1/zeros:output:0Pmodel_9/transformer_encoder_27/Encoder-2nd-NormalizationLayer-1/Const_2:output:0Pmodel_9/transformer_encoder_27/Encoder-2nd-NormalizationLayer-1/Const_3:output:0*
T0*
U0*x
_output_shapesf
d:"������������������:���������:���������:���������:���������:*
data_formatNCHW*
epsilon%o�:�
Imodel_9/transformer_encoder_27/Encoder-2nd-NormalizationLayer-1/Reshape_1ReshapeTmodel_9/transformer_encoder_27/Encoder-2nd-NormalizationLayer-1/FusedBatchNormV3:y:0Nmodel_9/transformer_encoder_27/Encoder-2nd-NormalizationLayer-1/Shape:output:0*
T0*+
_output_shapes
:���������P	�
Rmodel_9/transformer_encoder_27/Encoder-2nd-NormalizationLayer-1/mul/ReadVariableOpReadVariableOp[model_9_transformer_encoder_27_encoder_2nd_normalizationlayer_1_mul_readvariableop_resource*
_output_shapes
:	*
dtype0�
Cmodel_9/transformer_encoder_27/Encoder-2nd-NormalizationLayer-1/mulMulRmodel_9/transformer_encoder_27/Encoder-2nd-NormalizationLayer-1/Reshape_1:output:0Zmodel_9/transformer_encoder_27/Encoder-2nd-NormalizationLayer-1/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������P	�
Rmodel_9/transformer_encoder_27/Encoder-2nd-NormalizationLayer-1/add/ReadVariableOpReadVariableOp[model_9_transformer_encoder_27_encoder_2nd_normalizationlayer_1_add_readvariableop_resource*
_output_shapes
:	*
dtype0�
Cmodel_9/transformer_encoder_27/Encoder-2nd-NormalizationLayer-1/addAddV2Gmodel_9/transformer_encoder_27/Encoder-2nd-NormalizationLayer-1/mul:z:0Zmodel_9/transformer_encoder_27/Encoder-2nd-NormalizationLayer-1/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������P	�
Tmodel_9/transformer_encoder_27/Encoder-FeedForwardLayer_1_1/Tensordot/ReadVariableOpReadVariableOp]model_9_transformer_encoder_27_encoder_feedforwardlayer_1_1_tensordot_readvariableop_resource*
_output_shapes

:		*
dtype0�
Jmodel_9/transformer_encoder_27/Encoder-FeedForwardLayer_1_1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:�
Jmodel_9/transformer_encoder_27/Encoder-FeedForwardLayer_1_1/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       �
Kmodel_9/transformer_encoder_27/Encoder-FeedForwardLayer_1_1/Tensordot/ShapeShapeGmodel_9/transformer_encoder_27/Encoder-2nd-NormalizationLayer-1/add:z:0*
T0*
_output_shapes
::���
Smodel_9/transformer_encoder_27/Encoder-FeedForwardLayer_1_1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Nmodel_9/transformer_encoder_27/Encoder-FeedForwardLayer_1_1/Tensordot/GatherV2GatherV2Tmodel_9/transformer_encoder_27/Encoder-FeedForwardLayer_1_1/Tensordot/Shape:output:0Smodel_9/transformer_encoder_27/Encoder-FeedForwardLayer_1_1/Tensordot/free:output:0\model_9/transformer_encoder_27/Encoder-FeedForwardLayer_1_1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
Umodel_9/transformer_encoder_27/Encoder-FeedForwardLayer_1_1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Pmodel_9/transformer_encoder_27/Encoder-FeedForwardLayer_1_1/Tensordot/GatherV2_1GatherV2Tmodel_9/transformer_encoder_27/Encoder-FeedForwardLayer_1_1/Tensordot/Shape:output:0Smodel_9/transformer_encoder_27/Encoder-FeedForwardLayer_1_1/Tensordot/axes:output:0^model_9/transformer_encoder_27/Encoder-FeedForwardLayer_1_1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
Kmodel_9/transformer_encoder_27/Encoder-FeedForwardLayer_1_1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
Jmodel_9/transformer_encoder_27/Encoder-FeedForwardLayer_1_1/Tensordot/ProdProdWmodel_9/transformer_encoder_27/Encoder-FeedForwardLayer_1_1/Tensordot/GatherV2:output:0Tmodel_9/transformer_encoder_27/Encoder-FeedForwardLayer_1_1/Tensordot/Const:output:0*
T0*
_output_shapes
: �
Mmodel_9/transformer_encoder_27/Encoder-FeedForwardLayer_1_1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
Lmodel_9/transformer_encoder_27/Encoder-FeedForwardLayer_1_1/Tensordot/Prod_1ProdYmodel_9/transformer_encoder_27/Encoder-FeedForwardLayer_1_1/Tensordot/GatherV2_1:output:0Vmodel_9/transformer_encoder_27/Encoder-FeedForwardLayer_1_1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: �
Qmodel_9/transformer_encoder_27/Encoder-FeedForwardLayer_1_1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Lmodel_9/transformer_encoder_27/Encoder-FeedForwardLayer_1_1/Tensordot/concatConcatV2Smodel_9/transformer_encoder_27/Encoder-FeedForwardLayer_1_1/Tensordot/free:output:0Smodel_9/transformer_encoder_27/Encoder-FeedForwardLayer_1_1/Tensordot/axes:output:0Zmodel_9/transformer_encoder_27/Encoder-FeedForwardLayer_1_1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
Kmodel_9/transformer_encoder_27/Encoder-FeedForwardLayer_1_1/Tensordot/stackPackSmodel_9/transformer_encoder_27/Encoder-FeedForwardLayer_1_1/Tensordot/Prod:output:0Umodel_9/transformer_encoder_27/Encoder-FeedForwardLayer_1_1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
Omodel_9/transformer_encoder_27/Encoder-FeedForwardLayer_1_1/Tensordot/transpose	TransposeGmodel_9/transformer_encoder_27/Encoder-2nd-NormalizationLayer-1/add:z:0Umodel_9/transformer_encoder_27/Encoder-FeedForwardLayer_1_1/Tensordot/concat:output:0*
T0*+
_output_shapes
:���������P	�
Mmodel_9/transformer_encoder_27/Encoder-FeedForwardLayer_1_1/Tensordot/ReshapeReshapeSmodel_9/transformer_encoder_27/Encoder-FeedForwardLayer_1_1/Tensordot/transpose:y:0Tmodel_9/transformer_encoder_27/Encoder-FeedForwardLayer_1_1/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
Lmodel_9/transformer_encoder_27/Encoder-FeedForwardLayer_1_1/Tensordot/MatMulMatMulVmodel_9/transformer_encoder_27/Encoder-FeedForwardLayer_1_1/Tensordot/Reshape:output:0\model_9/transformer_encoder_27/Encoder-FeedForwardLayer_1_1/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������	�
Mmodel_9/transformer_encoder_27/Encoder-FeedForwardLayer_1_1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:	�
Smodel_9/transformer_encoder_27/Encoder-FeedForwardLayer_1_1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Nmodel_9/transformer_encoder_27/Encoder-FeedForwardLayer_1_1/Tensordot/concat_1ConcatV2Wmodel_9/transformer_encoder_27/Encoder-FeedForwardLayer_1_1/Tensordot/GatherV2:output:0Vmodel_9/transformer_encoder_27/Encoder-FeedForwardLayer_1_1/Tensordot/Const_2:output:0\model_9/transformer_encoder_27/Encoder-FeedForwardLayer_1_1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
Emodel_9/transformer_encoder_27/Encoder-FeedForwardLayer_1_1/TensordotReshapeVmodel_9/transformer_encoder_27/Encoder-FeedForwardLayer_1_1/Tensordot/MatMul:product:0Wmodel_9/transformer_encoder_27/Encoder-FeedForwardLayer_1_1/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������P	�
Rmodel_9/transformer_encoder_27/Encoder-FeedForwardLayer_1_1/BiasAdd/ReadVariableOpReadVariableOp[model_9_transformer_encoder_27_encoder_feedforwardlayer_1_1_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype0�
Cmodel_9/transformer_encoder_27/Encoder-FeedForwardLayer_1_1/BiasAddBiasAddNmodel_9/transformer_encoder_27/Encoder-FeedForwardLayer_1_1/Tensordot:output:0Zmodel_9/transformer_encoder_27/Encoder-FeedForwardLayer_1_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������P	�
Fmodel_9/transformer_encoder_27/Encoder-FeedForwardLayer_1_1/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
Dmodel_9/transformer_encoder_27/Encoder-FeedForwardLayer_1_1/Gelu/mulMulOmodel_9/transformer_encoder_27/Encoder-FeedForwardLayer_1_1/Gelu/mul/x:output:0Lmodel_9/transformer_encoder_27/Encoder-FeedForwardLayer_1_1/BiasAdd:output:0*
T0*+
_output_shapes
:���������P	�
Gmodel_9/transformer_encoder_27/Encoder-FeedForwardLayer_1_1/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *��?�
Hmodel_9/transformer_encoder_27/Encoder-FeedForwardLayer_1_1/Gelu/truedivRealDivLmodel_9/transformer_encoder_27/Encoder-FeedForwardLayer_1_1/BiasAdd:output:0Pmodel_9/transformer_encoder_27/Encoder-FeedForwardLayer_1_1/Gelu/Cast/x:output:0*
T0*+
_output_shapes
:���������P	�
Dmodel_9/transformer_encoder_27/Encoder-FeedForwardLayer_1_1/Gelu/ErfErfLmodel_9/transformer_encoder_27/Encoder-FeedForwardLayer_1_1/Gelu/truediv:z:0*
T0*+
_output_shapes
:���������P	�
Fmodel_9/transformer_encoder_27/Encoder-FeedForwardLayer_1_1/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
Dmodel_9/transformer_encoder_27/Encoder-FeedForwardLayer_1_1/Gelu/addAddV2Omodel_9/transformer_encoder_27/Encoder-FeedForwardLayer_1_1/Gelu/add/x:output:0Hmodel_9/transformer_encoder_27/Encoder-FeedForwardLayer_1_1/Gelu/Erf:y:0*
T0*+
_output_shapes
:���������P	�
Fmodel_9/transformer_encoder_27/Encoder-FeedForwardLayer_1_1/Gelu/mul_1MulHmodel_9/transformer_encoder_27/Encoder-FeedForwardLayer_1_1/Gelu/mul:z:0Hmodel_9/transformer_encoder_27/Encoder-FeedForwardLayer_1_1/Gelu/add:z:0*
T0*+
_output_shapes
:���������P	�
Tmodel_9/transformer_encoder_27/Encoder-FeedForwardLayer_2_1/Tensordot/ReadVariableOpReadVariableOp]model_9_transformer_encoder_27_encoder_feedforwardlayer_2_1_tensordot_readvariableop_resource*
_output_shapes

:		*
dtype0�
Jmodel_9/transformer_encoder_27/Encoder-FeedForwardLayer_2_1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:�
Jmodel_9/transformer_encoder_27/Encoder-FeedForwardLayer_2_1/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       �
Kmodel_9/transformer_encoder_27/Encoder-FeedForwardLayer_2_1/Tensordot/ShapeShapeJmodel_9/transformer_encoder_27/Encoder-FeedForwardLayer_1_1/Gelu/mul_1:z:0*
T0*
_output_shapes
::���
Smodel_9/transformer_encoder_27/Encoder-FeedForwardLayer_2_1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Nmodel_9/transformer_encoder_27/Encoder-FeedForwardLayer_2_1/Tensordot/GatherV2GatherV2Tmodel_9/transformer_encoder_27/Encoder-FeedForwardLayer_2_1/Tensordot/Shape:output:0Smodel_9/transformer_encoder_27/Encoder-FeedForwardLayer_2_1/Tensordot/free:output:0\model_9/transformer_encoder_27/Encoder-FeedForwardLayer_2_1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
Umodel_9/transformer_encoder_27/Encoder-FeedForwardLayer_2_1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Pmodel_9/transformer_encoder_27/Encoder-FeedForwardLayer_2_1/Tensordot/GatherV2_1GatherV2Tmodel_9/transformer_encoder_27/Encoder-FeedForwardLayer_2_1/Tensordot/Shape:output:0Smodel_9/transformer_encoder_27/Encoder-FeedForwardLayer_2_1/Tensordot/axes:output:0^model_9/transformer_encoder_27/Encoder-FeedForwardLayer_2_1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
Kmodel_9/transformer_encoder_27/Encoder-FeedForwardLayer_2_1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
Jmodel_9/transformer_encoder_27/Encoder-FeedForwardLayer_2_1/Tensordot/ProdProdWmodel_9/transformer_encoder_27/Encoder-FeedForwardLayer_2_1/Tensordot/GatherV2:output:0Tmodel_9/transformer_encoder_27/Encoder-FeedForwardLayer_2_1/Tensordot/Const:output:0*
T0*
_output_shapes
: �
Mmodel_9/transformer_encoder_27/Encoder-FeedForwardLayer_2_1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
Lmodel_9/transformer_encoder_27/Encoder-FeedForwardLayer_2_1/Tensordot/Prod_1ProdYmodel_9/transformer_encoder_27/Encoder-FeedForwardLayer_2_1/Tensordot/GatherV2_1:output:0Vmodel_9/transformer_encoder_27/Encoder-FeedForwardLayer_2_1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: �
Qmodel_9/transformer_encoder_27/Encoder-FeedForwardLayer_2_1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Lmodel_9/transformer_encoder_27/Encoder-FeedForwardLayer_2_1/Tensordot/concatConcatV2Smodel_9/transformer_encoder_27/Encoder-FeedForwardLayer_2_1/Tensordot/free:output:0Smodel_9/transformer_encoder_27/Encoder-FeedForwardLayer_2_1/Tensordot/axes:output:0Zmodel_9/transformer_encoder_27/Encoder-FeedForwardLayer_2_1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
Kmodel_9/transformer_encoder_27/Encoder-FeedForwardLayer_2_1/Tensordot/stackPackSmodel_9/transformer_encoder_27/Encoder-FeedForwardLayer_2_1/Tensordot/Prod:output:0Umodel_9/transformer_encoder_27/Encoder-FeedForwardLayer_2_1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
Omodel_9/transformer_encoder_27/Encoder-FeedForwardLayer_2_1/Tensordot/transpose	TransposeJmodel_9/transformer_encoder_27/Encoder-FeedForwardLayer_1_1/Gelu/mul_1:z:0Umodel_9/transformer_encoder_27/Encoder-FeedForwardLayer_2_1/Tensordot/concat:output:0*
T0*+
_output_shapes
:���������P	�
Mmodel_9/transformer_encoder_27/Encoder-FeedForwardLayer_2_1/Tensordot/ReshapeReshapeSmodel_9/transformer_encoder_27/Encoder-FeedForwardLayer_2_1/Tensordot/transpose:y:0Tmodel_9/transformer_encoder_27/Encoder-FeedForwardLayer_2_1/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
Lmodel_9/transformer_encoder_27/Encoder-FeedForwardLayer_2_1/Tensordot/MatMulMatMulVmodel_9/transformer_encoder_27/Encoder-FeedForwardLayer_2_1/Tensordot/Reshape:output:0\model_9/transformer_encoder_27/Encoder-FeedForwardLayer_2_1/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������	�
Mmodel_9/transformer_encoder_27/Encoder-FeedForwardLayer_2_1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:	�
Smodel_9/transformer_encoder_27/Encoder-FeedForwardLayer_2_1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Nmodel_9/transformer_encoder_27/Encoder-FeedForwardLayer_2_1/Tensordot/concat_1ConcatV2Wmodel_9/transformer_encoder_27/Encoder-FeedForwardLayer_2_1/Tensordot/GatherV2:output:0Vmodel_9/transformer_encoder_27/Encoder-FeedForwardLayer_2_1/Tensordot/Const_2:output:0\model_9/transformer_encoder_27/Encoder-FeedForwardLayer_2_1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
Emodel_9/transformer_encoder_27/Encoder-FeedForwardLayer_2_1/TensordotReshapeVmodel_9/transformer_encoder_27/Encoder-FeedForwardLayer_2_1/Tensordot/MatMul:product:0Wmodel_9/transformer_encoder_27/Encoder-FeedForwardLayer_2_1/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������P	�
Rmodel_9/transformer_encoder_27/Encoder-FeedForwardLayer_2_1/BiasAdd/ReadVariableOpReadVariableOp[model_9_transformer_encoder_27_encoder_feedforwardlayer_2_1_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype0�
Cmodel_9/transformer_encoder_27/Encoder-FeedForwardLayer_2_1/BiasAddBiasAddNmodel_9/transformer_encoder_27/Encoder-FeedForwardLayer_2_1/Tensordot:output:0Zmodel_9/transformer_encoder_27/Encoder-FeedForwardLayer_2_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������P	�
2model_9/transformer_encoder_27/dropout_27/IdentityIdentityLmodel_9/transformer_encoder_27/Encoder-FeedForwardLayer_2_1/BiasAdd:output:0*
T0*+
_output_shapes
:���������P	�
>model_9/transformer_encoder_27/Encoder-2nd-AdditionLayer-1/addAddV2Bmodel_9/transformer_encoder_27/Encoder-1st-AdditionLayer-1/add:z:0;model_9/transformer_encoder_27/dropout_27/Identity:output:0*
T0*+
_output_shapes
:���������P	�
Emodel_9/transformer_encoder_28/Encoder-1st-NormalizationLayer-2/ShapeShapeBmodel_9/transformer_encoder_27/Encoder-2nd-AdditionLayer-1/add:z:0*
T0*
_output_shapes
::���
Smodel_9/transformer_encoder_28/Encoder-1st-NormalizationLayer-2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
Umodel_9/transformer_encoder_28/Encoder-1st-NormalizationLayer-2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
Umodel_9/transformer_encoder_28/Encoder-1st-NormalizationLayer-2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
Mmodel_9/transformer_encoder_28/Encoder-1st-NormalizationLayer-2/strided_sliceStridedSliceNmodel_9/transformer_encoder_28/Encoder-1st-NormalizationLayer-2/Shape:output:0\model_9/transformer_encoder_28/Encoder-1st-NormalizationLayer-2/strided_slice/stack:output:0^model_9/transformer_encoder_28/Encoder-1st-NormalizationLayer-2/strided_slice/stack_1:output:0^model_9/transformer_encoder_28/Encoder-1st-NormalizationLayer-2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask�
Emodel_9/transformer_encoder_28/Encoder-1st-NormalizationLayer-2/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
Dmodel_9/transformer_encoder_28/Encoder-1st-NormalizationLayer-2/ProdProdVmodel_9/transformer_encoder_28/Encoder-1st-NormalizationLayer-2/strided_slice:output:0Nmodel_9/transformer_encoder_28/Encoder-1st-NormalizationLayer-2/Const:output:0*
T0*
_output_shapes
: �
Umodel_9/transformer_encoder_28/Encoder-1st-NormalizationLayer-2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:�
Wmodel_9/transformer_encoder_28/Encoder-1st-NormalizationLayer-2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: �
Wmodel_9/transformer_encoder_28/Encoder-1st-NormalizationLayer-2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
Omodel_9/transformer_encoder_28/Encoder-1st-NormalizationLayer-2/strided_slice_1StridedSliceNmodel_9/transformer_encoder_28/Encoder-1st-NormalizationLayer-2/Shape:output:0^model_9/transformer_encoder_28/Encoder-1st-NormalizationLayer-2/strided_slice_1/stack:output:0`model_9/transformer_encoder_28/Encoder-1st-NormalizationLayer-2/strided_slice_1/stack_1:output:0`model_9/transformer_encoder_28/Encoder-1st-NormalizationLayer-2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask�
Gmodel_9/transformer_encoder_28/Encoder-1st-NormalizationLayer-2/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
Fmodel_9/transformer_encoder_28/Encoder-1st-NormalizationLayer-2/Prod_1ProdXmodel_9/transformer_encoder_28/Encoder-1st-NormalizationLayer-2/strided_slice_1:output:0Pmodel_9/transformer_encoder_28/Encoder-1st-NormalizationLayer-2/Const_1:output:0*
T0*
_output_shapes
: �
Omodel_9/transformer_encoder_28/Encoder-1st-NormalizationLayer-2/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :�
Omodel_9/transformer_encoder_28/Encoder-1st-NormalizationLayer-2/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :�
Mmodel_9/transformer_encoder_28/Encoder-1st-NormalizationLayer-2/Reshape/shapePackXmodel_9/transformer_encoder_28/Encoder-1st-NormalizationLayer-2/Reshape/shape/0:output:0Mmodel_9/transformer_encoder_28/Encoder-1st-NormalizationLayer-2/Prod:output:0Omodel_9/transformer_encoder_28/Encoder-1st-NormalizationLayer-2/Prod_1:output:0Xmodel_9/transformer_encoder_28/Encoder-1st-NormalizationLayer-2/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:�
Gmodel_9/transformer_encoder_28/Encoder-1st-NormalizationLayer-2/ReshapeReshapeBmodel_9/transformer_encoder_27/Encoder-2nd-AdditionLayer-1/add:z:0Vmodel_9/transformer_encoder_28/Encoder-1st-NormalizationLayer-2/Reshape/shape:output:0*
T0*8
_output_shapes&
$:"�������������������
Kmodel_9/transformer_encoder_28/Encoder-1st-NormalizationLayer-2/ones/packedPackMmodel_9/transformer_encoder_28/Encoder-1st-NormalizationLayer-2/Prod:output:0*
N*
T0*
_output_shapes
:�
Jmodel_9/transformer_encoder_28/Encoder-1st-NormalizationLayer-2/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
Dmodel_9/transformer_encoder_28/Encoder-1st-NormalizationLayer-2/onesFillTmodel_9/transformer_encoder_28/Encoder-1st-NormalizationLayer-2/ones/packed:output:0Smodel_9/transformer_encoder_28/Encoder-1st-NormalizationLayer-2/ones/Const:output:0*
T0*#
_output_shapes
:����������
Lmodel_9/transformer_encoder_28/Encoder-1st-NormalizationLayer-2/zeros/packedPackMmodel_9/transformer_encoder_28/Encoder-1st-NormalizationLayer-2/Prod:output:0*
N*
T0*
_output_shapes
:�
Kmodel_9/transformer_encoder_28/Encoder-1st-NormalizationLayer-2/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
Emodel_9/transformer_encoder_28/Encoder-1st-NormalizationLayer-2/zerosFillUmodel_9/transformer_encoder_28/Encoder-1st-NormalizationLayer-2/zeros/packed:output:0Tmodel_9/transformer_encoder_28/Encoder-1st-NormalizationLayer-2/zeros/Const:output:0*
T0*#
_output_shapes
:����������
Gmodel_9/transformer_encoder_28/Encoder-1st-NormalizationLayer-2/Const_2Const*
_output_shapes
: *
dtype0*
valueB �
Gmodel_9/transformer_encoder_28/Encoder-1st-NormalizationLayer-2/Const_3Const*
_output_shapes
: *
dtype0*
valueB �
Pmodel_9/transformer_encoder_28/Encoder-1st-NormalizationLayer-2/FusedBatchNormV3FusedBatchNormV3Pmodel_9/transformer_encoder_28/Encoder-1st-NormalizationLayer-2/Reshape:output:0Mmodel_9/transformer_encoder_28/Encoder-1st-NormalizationLayer-2/ones:output:0Nmodel_9/transformer_encoder_28/Encoder-1st-NormalizationLayer-2/zeros:output:0Pmodel_9/transformer_encoder_28/Encoder-1st-NormalizationLayer-2/Const_2:output:0Pmodel_9/transformer_encoder_28/Encoder-1st-NormalizationLayer-2/Const_3:output:0*
T0*
U0*x
_output_shapesf
d:"������������������:���������:���������:���������:���������:*
data_formatNCHW*
epsilon%o�:�
Imodel_9/transformer_encoder_28/Encoder-1st-NormalizationLayer-2/Reshape_1ReshapeTmodel_9/transformer_encoder_28/Encoder-1st-NormalizationLayer-2/FusedBatchNormV3:y:0Nmodel_9/transformer_encoder_28/Encoder-1st-NormalizationLayer-2/Shape:output:0*
T0*+
_output_shapes
:���������P	�
Rmodel_9/transformer_encoder_28/Encoder-1st-NormalizationLayer-2/mul/ReadVariableOpReadVariableOp[model_9_transformer_encoder_28_encoder_1st_normalizationlayer_2_mul_readvariableop_resource*
_output_shapes
:	*
dtype0�
Cmodel_9/transformer_encoder_28/Encoder-1st-NormalizationLayer-2/mulMulRmodel_9/transformer_encoder_28/Encoder-1st-NormalizationLayer-2/Reshape_1:output:0Zmodel_9/transformer_encoder_28/Encoder-1st-NormalizationLayer-2/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������P	�
Rmodel_9/transformer_encoder_28/Encoder-1st-NormalizationLayer-2/add/ReadVariableOpReadVariableOp[model_9_transformer_encoder_28_encoder_1st_normalizationlayer_2_add_readvariableop_resource*
_output_shapes
:	*
dtype0�
Cmodel_9/transformer_encoder_28/Encoder-1st-NormalizationLayer-2/addAddV2Gmodel_9/transformer_encoder_28/Encoder-1st-NormalizationLayer-2/mul:z:0Zmodel_9/transformer_encoder_28/Encoder-1st-NormalizationLayer-2/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������P	�
^model_9/transformer_encoder_28/Encoder-SelfAttentionLayer-2/query/einsum/Einsum/ReadVariableOpReadVariableOpgmodel_9_transformer_encoder_28_encoder_selfattentionlayer_2_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:	*
dtype0�
Omodel_9/transformer_encoder_28/Encoder-SelfAttentionLayer-2/query/einsum/EinsumEinsumGmodel_9/transformer_encoder_28/Encoder-1st-NormalizationLayer-2/add:z:0fmodel_9/transformer_encoder_28/Encoder-SelfAttentionLayer-2/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������P*
equationabc,cde->abde�
Tmodel_9/transformer_encoder_28/Encoder-SelfAttentionLayer-2/query/add/ReadVariableOpReadVariableOp]model_9_transformer_encoder_28_encoder_selfattentionlayer_2_query_add_readvariableop_resource*
_output_shapes

:*
dtype0�
Emodel_9/transformer_encoder_28/Encoder-SelfAttentionLayer-2/query/addAddV2Xmodel_9/transformer_encoder_28/Encoder-SelfAttentionLayer-2/query/einsum/Einsum:output:0\model_9/transformer_encoder_28/Encoder-SelfAttentionLayer-2/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������P�
\model_9/transformer_encoder_28/Encoder-SelfAttentionLayer-2/key/einsum/Einsum/ReadVariableOpReadVariableOpemodel_9_transformer_encoder_28_encoder_selfattentionlayer_2_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:	*
dtype0�
Mmodel_9/transformer_encoder_28/Encoder-SelfAttentionLayer-2/key/einsum/EinsumEinsumGmodel_9/transformer_encoder_28/Encoder-1st-NormalizationLayer-2/add:z:0dmodel_9/transformer_encoder_28/Encoder-SelfAttentionLayer-2/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������P*
equationabc,cde->abde�
Rmodel_9/transformer_encoder_28/Encoder-SelfAttentionLayer-2/key/add/ReadVariableOpReadVariableOp[model_9_transformer_encoder_28_encoder_selfattentionlayer_2_key_add_readvariableop_resource*
_output_shapes

:*
dtype0�
Cmodel_9/transformer_encoder_28/Encoder-SelfAttentionLayer-2/key/addAddV2Vmodel_9/transformer_encoder_28/Encoder-SelfAttentionLayer-2/key/einsum/Einsum:output:0Zmodel_9/transformer_encoder_28/Encoder-SelfAttentionLayer-2/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������P�
^model_9/transformer_encoder_28/Encoder-SelfAttentionLayer-2/value/einsum/Einsum/ReadVariableOpReadVariableOpgmodel_9_transformer_encoder_28_encoder_selfattentionlayer_2_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:	*
dtype0�
Omodel_9/transformer_encoder_28/Encoder-SelfAttentionLayer-2/value/einsum/EinsumEinsumGmodel_9/transformer_encoder_28/Encoder-1st-NormalizationLayer-2/add:z:0fmodel_9/transformer_encoder_28/Encoder-SelfAttentionLayer-2/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������P*
equationabc,cde->abde�
Tmodel_9/transformer_encoder_28/Encoder-SelfAttentionLayer-2/value/add/ReadVariableOpReadVariableOp]model_9_transformer_encoder_28_encoder_selfattentionlayer_2_value_add_readvariableop_resource*
_output_shapes

:*
dtype0�
Emodel_9/transformer_encoder_28/Encoder-SelfAttentionLayer-2/value/addAddV2Xmodel_9/transformer_encoder_28/Encoder-SelfAttentionLayer-2/value/einsum/Einsum:output:0\model_9/transformer_encoder_28/Encoder-SelfAttentionLayer-2/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������P�
Amodel_9/transformer_encoder_28/Encoder-SelfAttentionLayer-2/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *:�?�
?model_9/transformer_encoder_28/Encoder-SelfAttentionLayer-2/MulMulImodel_9/transformer_encoder_28/Encoder-SelfAttentionLayer-2/query/add:z:0Jmodel_9/transformer_encoder_28/Encoder-SelfAttentionLayer-2/Mul/y:output:0*
T0*/
_output_shapes
:���������P�
Imodel_9/transformer_encoder_28/Encoder-SelfAttentionLayer-2/einsum/EinsumEinsumGmodel_9/transformer_encoder_28/Encoder-SelfAttentionLayer-2/key/add:z:0Cmodel_9/transformer_encoder_28/Encoder-SelfAttentionLayer-2/Mul:z:0*
N*
T0*/
_output_shapes
:���������PP*
equationaecd,abcd->acbe�
Kmodel_9/transformer_encoder_28/Encoder-SelfAttentionLayer-2/softmax/SoftmaxSoftmaxRmodel_9/transformer_encoder_28/Encoder-SelfAttentionLayer-2/einsum/Einsum:output:0*
T0*/
_output_shapes
:���������PP�
Lmodel_9/transformer_encoder_28/Encoder-SelfAttentionLayer-2/dropout/IdentityIdentityUmodel_9/transformer_encoder_28/Encoder-SelfAttentionLayer-2/softmax/Softmax:softmax:0*
T0*/
_output_shapes
:���������PP�
Kmodel_9/transformer_encoder_28/Encoder-SelfAttentionLayer-2/einsum_1/EinsumEinsumUmodel_9/transformer_encoder_28/Encoder-SelfAttentionLayer-2/dropout/Identity:output:0Imodel_9/transformer_encoder_28/Encoder-SelfAttentionLayer-2/value/add:z:0*
N*
T0*/
_output_shapes
:���������P*
equationacbe,aecd->abcd�
imodel_9/transformer_encoder_28/Encoder-SelfAttentionLayer-2/attention_output/einsum/Einsum/ReadVariableOpReadVariableOprmodel_9_transformer_encoder_28_encoder_selfattentionlayer_2_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:	*
dtype0�
Zmodel_9/transformer_encoder_28/Encoder-SelfAttentionLayer-2/attention_output/einsum/EinsumEinsumTmodel_9/transformer_encoder_28/Encoder-SelfAttentionLayer-2/einsum_1/Einsum:output:0qmodel_9/transformer_encoder_28/Encoder-SelfAttentionLayer-2/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:���������P	*
equationabcd,cde->abe�
_model_9/transformer_encoder_28/Encoder-SelfAttentionLayer-2/attention_output/add/ReadVariableOpReadVariableOphmodel_9_transformer_encoder_28_encoder_selfattentionlayer_2_attention_output_add_readvariableop_resource*
_output_shapes
:	*
dtype0�
Pmodel_9/transformer_encoder_28/Encoder-SelfAttentionLayer-2/attention_output/addAddV2cmodel_9/transformer_encoder_28/Encoder-SelfAttentionLayer-2/attention_output/einsum/Einsum:output:0gmodel_9/transformer_encoder_28/Encoder-SelfAttentionLayer-2/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������P	�
>model_9/transformer_encoder_28/Encoder-1st-AdditionLayer-2/addAddV2Bmodel_9/transformer_encoder_27/Encoder-2nd-AdditionLayer-1/add:z:0Tmodel_9/transformer_encoder_28/Encoder-SelfAttentionLayer-2/attention_output/add:z:0*
T0*+
_output_shapes
:���������P	�
Emodel_9/transformer_encoder_28/Encoder-2nd-NormalizationLayer-2/ShapeShapeBmodel_9/transformer_encoder_28/Encoder-1st-AdditionLayer-2/add:z:0*
T0*
_output_shapes
::���
Smodel_9/transformer_encoder_28/Encoder-2nd-NormalizationLayer-2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
Umodel_9/transformer_encoder_28/Encoder-2nd-NormalizationLayer-2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
Umodel_9/transformer_encoder_28/Encoder-2nd-NormalizationLayer-2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
Mmodel_9/transformer_encoder_28/Encoder-2nd-NormalizationLayer-2/strided_sliceStridedSliceNmodel_9/transformer_encoder_28/Encoder-2nd-NormalizationLayer-2/Shape:output:0\model_9/transformer_encoder_28/Encoder-2nd-NormalizationLayer-2/strided_slice/stack:output:0^model_9/transformer_encoder_28/Encoder-2nd-NormalizationLayer-2/strided_slice/stack_1:output:0^model_9/transformer_encoder_28/Encoder-2nd-NormalizationLayer-2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask�
Emodel_9/transformer_encoder_28/Encoder-2nd-NormalizationLayer-2/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
Dmodel_9/transformer_encoder_28/Encoder-2nd-NormalizationLayer-2/ProdProdVmodel_9/transformer_encoder_28/Encoder-2nd-NormalizationLayer-2/strided_slice:output:0Nmodel_9/transformer_encoder_28/Encoder-2nd-NormalizationLayer-2/Const:output:0*
T0*
_output_shapes
: �
Umodel_9/transformer_encoder_28/Encoder-2nd-NormalizationLayer-2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:�
Wmodel_9/transformer_encoder_28/Encoder-2nd-NormalizationLayer-2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: �
Wmodel_9/transformer_encoder_28/Encoder-2nd-NormalizationLayer-2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
Omodel_9/transformer_encoder_28/Encoder-2nd-NormalizationLayer-2/strided_slice_1StridedSliceNmodel_9/transformer_encoder_28/Encoder-2nd-NormalizationLayer-2/Shape:output:0^model_9/transformer_encoder_28/Encoder-2nd-NormalizationLayer-2/strided_slice_1/stack:output:0`model_9/transformer_encoder_28/Encoder-2nd-NormalizationLayer-2/strided_slice_1/stack_1:output:0`model_9/transformer_encoder_28/Encoder-2nd-NormalizationLayer-2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask�
Gmodel_9/transformer_encoder_28/Encoder-2nd-NormalizationLayer-2/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
Fmodel_9/transformer_encoder_28/Encoder-2nd-NormalizationLayer-2/Prod_1ProdXmodel_9/transformer_encoder_28/Encoder-2nd-NormalizationLayer-2/strided_slice_1:output:0Pmodel_9/transformer_encoder_28/Encoder-2nd-NormalizationLayer-2/Const_1:output:0*
T0*
_output_shapes
: �
Omodel_9/transformer_encoder_28/Encoder-2nd-NormalizationLayer-2/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :�
Omodel_9/transformer_encoder_28/Encoder-2nd-NormalizationLayer-2/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :�
Mmodel_9/transformer_encoder_28/Encoder-2nd-NormalizationLayer-2/Reshape/shapePackXmodel_9/transformer_encoder_28/Encoder-2nd-NormalizationLayer-2/Reshape/shape/0:output:0Mmodel_9/transformer_encoder_28/Encoder-2nd-NormalizationLayer-2/Prod:output:0Omodel_9/transformer_encoder_28/Encoder-2nd-NormalizationLayer-2/Prod_1:output:0Xmodel_9/transformer_encoder_28/Encoder-2nd-NormalizationLayer-2/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:�
Gmodel_9/transformer_encoder_28/Encoder-2nd-NormalizationLayer-2/ReshapeReshapeBmodel_9/transformer_encoder_28/Encoder-1st-AdditionLayer-2/add:z:0Vmodel_9/transformer_encoder_28/Encoder-2nd-NormalizationLayer-2/Reshape/shape:output:0*
T0*8
_output_shapes&
$:"�������������������
Kmodel_9/transformer_encoder_28/Encoder-2nd-NormalizationLayer-2/ones/packedPackMmodel_9/transformer_encoder_28/Encoder-2nd-NormalizationLayer-2/Prod:output:0*
N*
T0*
_output_shapes
:�
Jmodel_9/transformer_encoder_28/Encoder-2nd-NormalizationLayer-2/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
Dmodel_9/transformer_encoder_28/Encoder-2nd-NormalizationLayer-2/onesFillTmodel_9/transformer_encoder_28/Encoder-2nd-NormalizationLayer-2/ones/packed:output:0Smodel_9/transformer_encoder_28/Encoder-2nd-NormalizationLayer-2/ones/Const:output:0*
T0*#
_output_shapes
:����������
Lmodel_9/transformer_encoder_28/Encoder-2nd-NormalizationLayer-2/zeros/packedPackMmodel_9/transformer_encoder_28/Encoder-2nd-NormalizationLayer-2/Prod:output:0*
N*
T0*
_output_shapes
:�
Kmodel_9/transformer_encoder_28/Encoder-2nd-NormalizationLayer-2/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
Emodel_9/transformer_encoder_28/Encoder-2nd-NormalizationLayer-2/zerosFillUmodel_9/transformer_encoder_28/Encoder-2nd-NormalizationLayer-2/zeros/packed:output:0Tmodel_9/transformer_encoder_28/Encoder-2nd-NormalizationLayer-2/zeros/Const:output:0*
T0*#
_output_shapes
:����������
Gmodel_9/transformer_encoder_28/Encoder-2nd-NormalizationLayer-2/Const_2Const*
_output_shapes
: *
dtype0*
valueB �
Gmodel_9/transformer_encoder_28/Encoder-2nd-NormalizationLayer-2/Const_3Const*
_output_shapes
: *
dtype0*
valueB �
Pmodel_9/transformer_encoder_28/Encoder-2nd-NormalizationLayer-2/FusedBatchNormV3FusedBatchNormV3Pmodel_9/transformer_encoder_28/Encoder-2nd-NormalizationLayer-2/Reshape:output:0Mmodel_9/transformer_encoder_28/Encoder-2nd-NormalizationLayer-2/ones:output:0Nmodel_9/transformer_encoder_28/Encoder-2nd-NormalizationLayer-2/zeros:output:0Pmodel_9/transformer_encoder_28/Encoder-2nd-NormalizationLayer-2/Const_2:output:0Pmodel_9/transformer_encoder_28/Encoder-2nd-NormalizationLayer-2/Const_3:output:0*
T0*
U0*x
_output_shapesf
d:"������������������:���������:���������:���������:���������:*
data_formatNCHW*
epsilon%o�:�
Imodel_9/transformer_encoder_28/Encoder-2nd-NormalizationLayer-2/Reshape_1ReshapeTmodel_9/transformer_encoder_28/Encoder-2nd-NormalizationLayer-2/FusedBatchNormV3:y:0Nmodel_9/transformer_encoder_28/Encoder-2nd-NormalizationLayer-2/Shape:output:0*
T0*+
_output_shapes
:���������P	�
Rmodel_9/transformer_encoder_28/Encoder-2nd-NormalizationLayer-2/mul/ReadVariableOpReadVariableOp[model_9_transformer_encoder_28_encoder_2nd_normalizationlayer_2_mul_readvariableop_resource*
_output_shapes
:	*
dtype0�
Cmodel_9/transformer_encoder_28/Encoder-2nd-NormalizationLayer-2/mulMulRmodel_9/transformer_encoder_28/Encoder-2nd-NormalizationLayer-2/Reshape_1:output:0Zmodel_9/transformer_encoder_28/Encoder-2nd-NormalizationLayer-2/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������P	�
Rmodel_9/transformer_encoder_28/Encoder-2nd-NormalizationLayer-2/add/ReadVariableOpReadVariableOp[model_9_transformer_encoder_28_encoder_2nd_normalizationlayer_2_add_readvariableop_resource*
_output_shapes
:	*
dtype0�
Cmodel_9/transformer_encoder_28/Encoder-2nd-NormalizationLayer-2/addAddV2Gmodel_9/transformer_encoder_28/Encoder-2nd-NormalizationLayer-2/mul:z:0Zmodel_9/transformer_encoder_28/Encoder-2nd-NormalizationLayer-2/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������P	�
Tmodel_9/transformer_encoder_28/Encoder-FeedForwardLayer_1_2/Tensordot/ReadVariableOpReadVariableOp]model_9_transformer_encoder_28_encoder_feedforwardlayer_1_2_tensordot_readvariableop_resource*
_output_shapes

:		*
dtype0�
Jmodel_9/transformer_encoder_28/Encoder-FeedForwardLayer_1_2/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:�
Jmodel_9/transformer_encoder_28/Encoder-FeedForwardLayer_1_2/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       �
Kmodel_9/transformer_encoder_28/Encoder-FeedForwardLayer_1_2/Tensordot/ShapeShapeGmodel_9/transformer_encoder_28/Encoder-2nd-NormalizationLayer-2/add:z:0*
T0*
_output_shapes
::���
Smodel_9/transformer_encoder_28/Encoder-FeedForwardLayer_1_2/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Nmodel_9/transformer_encoder_28/Encoder-FeedForwardLayer_1_2/Tensordot/GatherV2GatherV2Tmodel_9/transformer_encoder_28/Encoder-FeedForwardLayer_1_2/Tensordot/Shape:output:0Smodel_9/transformer_encoder_28/Encoder-FeedForwardLayer_1_2/Tensordot/free:output:0\model_9/transformer_encoder_28/Encoder-FeedForwardLayer_1_2/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
Umodel_9/transformer_encoder_28/Encoder-FeedForwardLayer_1_2/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Pmodel_9/transformer_encoder_28/Encoder-FeedForwardLayer_1_2/Tensordot/GatherV2_1GatherV2Tmodel_9/transformer_encoder_28/Encoder-FeedForwardLayer_1_2/Tensordot/Shape:output:0Smodel_9/transformer_encoder_28/Encoder-FeedForwardLayer_1_2/Tensordot/axes:output:0^model_9/transformer_encoder_28/Encoder-FeedForwardLayer_1_2/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
Kmodel_9/transformer_encoder_28/Encoder-FeedForwardLayer_1_2/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
Jmodel_9/transformer_encoder_28/Encoder-FeedForwardLayer_1_2/Tensordot/ProdProdWmodel_9/transformer_encoder_28/Encoder-FeedForwardLayer_1_2/Tensordot/GatherV2:output:0Tmodel_9/transformer_encoder_28/Encoder-FeedForwardLayer_1_2/Tensordot/Const:output:0*
T0*
_output_shapes
: �
Mmodel_9/transformer_encoder_28/Encoder-FeedForwardLayer_1_2/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
Lmodel_9/transformer_encoder_28/Encoder-FeedForwardLayer_1_2/Tensordot/Prod_1ProdYmodel_9/transformer_encoder_28/Encoder-FeedForwardLayer_1_2/Tensordot/GatherV2_1:output:0Vmodel_9/transformer_encoder_28/Encoder-FeedForwardLayer_1_2/Tensordot/Const_1:output:0*
T0*
_output_shapes
: �
Qmodel_9/transformer_encoder_28/Encoder-FeedForwardLayer_1_2/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Lmodel_9/transformer_encoder_28/Encoder-FeedForwardLayer_1_2/Tensordot/concatConcatV2Smodel_9/transformer_encoder_28/Encoder-FeedForwardLayer_1_2/Tensordot/free:output:0Smodel_9/transformer_encoder_28/Encoder-FeedForwardLayer_1_2/Tensordot/axes:output:0Zmodel_9/transformer_encoder_28/Encoder-FeedForwardLayer_1_2/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
Kmodel_9/transformer_encoder_28/Encoder-FeedForwardLayer_1_2/Tensordot/stackPackSmodel_9/transformer_encoder_28/Encoder-FeedForwardLayer_1_2/Tensordot/Prod:output:0Umodel_9/transformer_encoder_28/Encoder-FeedForwardLayer_1_2/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
Omodel_9/transformer_encoder_28/Encoder-FeedForwardLayer_1_2/Tensordot/transpose	TransposeGmodel_9/transformer_encoder_28/Encoder-2nd-NormalizationLayer-2/add:z:0Umodel_9/transformer_encoder_28/Encoder-FeedForwardLayer_1_2/Tensordot/concat:output:0*
T0*+
_output_shapes
:���������P	�
Mmodel_9/transformer_encoder_28/Encoder-FeedForwardLayer_1_2/Tensordot/ReshapeReshapeSmodel_9/transformer_encoder_28/Encoder-FeedForwardLayer_1_2/Tensordot/transpose:y:0Tmodel_9/transformer_encoder_28/Encoder-FeedForwardLayer_1_2/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
Lmodel_9/transformer_encoder_28/Encoder-FeedForwardLayer_1_2/Tensordot/MatMulMatMulVmodel_9/transformer_encoder_28/Encoder-FeedForwardLayer_1_2/Tensordot/Reshape:output:0\model_9/transformer_encoder_28/Encoder-FeedForwardLayer_1_2/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������	�
Mmodel_9/transformer_encoder_28/Encoder-FeedForwardLayer_1_2/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:	�
Smodel_9/transformer_encoder_28/Encoder-FeedForwardLayer_1_2/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Nmodel_9/transformer_encoder_28/Encoder-FeedForwardLayer_1_2/Tensordot/concat_1ConcatV2Wmodel_9/transformer_encoder_28/Encoder-FeedForwardLayer_1_2/Tensordot/GatherV2:output:0Vmodel_9/transformer_encoder_28/Encoder-FeedForwardLayer_1_2/Tensordot/Const_2:output:0\model_9/transformer_encoder_28/Encoder-FeedForwardLayer_1_2/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
Emodel_9/transformer_encoder_28/Encoder-FeedForwardLayer_1_2/TensordotReshapeVmodel_9/transformer_encoder_28/Encoder-FeedForwardLayer_1_2/Tensordot/MatMul:product:0Wmodel_9/transformer_encoder_28/Encoder-FeedForwardLayer_1_2/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������P	�
Rmodel_9/transformer_encoder_28/Encoder-FeedForwardLayer_1_2/BiasAdd/ReadVariableOpReadVariableOp[model_9_transformer_encoder_28_encoder_feedforwardlayer_1_2_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype0�
Cmodel_9/transformer_encoder_28/Encoder-FeedForwardLayer_1_2/BiasAddBiasAddNmodel_9/transformer_encoder_28/Encoder-FeedForwardLayer_1_2/Tensordot:output:0Zmodel_9/transformer_encoder_28/Encoder-FeedForwardLayer_1_2/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������P	�
Fmodel_9/transformer_encoder_28/Encoder-FeedForwardLayer_1_2/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
Dmodel_9/transformer_encoder_28/Encoder-FeedForwardLayer_1_2/Gelu/mulMulOmodel_9/transformer_encoder_28/Encoder-FeedForwardLayer_1_2/Gelu/mul/x:output:0Lmodel_9/transformer_encoder_28/Encoder-FeedForwardLayer_1_2/BiasAdd:output:0*
T0*+
_output_shapes
:���������P	�
Gmodel_9/transformer_encoder_28/Encoder-FeedForwardLayer_1_2/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *��?�
Hmodel_9/transformer_encoder_28/Encoder-FeedForwardLayer_1_2/Gelu/truedivRealDivLmodel_9/transformer_encoder_28/Encoder-FeedForwardLayer_1_2/BiasAdd:output:0Pmodel_9/transformer_encoder_28/Encoder-FeedForwardLayer_1_2/Gelu/Cast/x:output:0*
T0*+
_output_shapes
:���������P	�
Dmodel_9/transformer_encoder_28/Encoder-FeedForwardLayer_1_2/Gelu/ErfErfLmodel_9/transformer_encoder_28/Encoder-FeedForwardLayer_1_2/Gelu/truediv:z:0*
T0*+
_output_shapes
:���������P	�
Fmodel_9/transformer_encoder_28/Encoder-FeedForwardLayer_1_2/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
Dmodel_9/transformer_encoder_28/Encoder-FeedForwardLayer_1_2/Gelu/addAddV2Omodel_9/transformer_encoder_28/Encoder-FeedForwardLayer_1_2/Gelu/add/x:output:0Hmodel_9/transformer_encoder_28/Encoder-FeedForwardLayer_1_2/Gelu/Erf:y:0*
T0*+
_output_shapes
:���������P	�
Fmodel_9/transformer_encoder_28/Encoder-FeedForwardLayer_1_2/Gelu/mul_1MulHmodel_9/transformer_encoder_28/Encoder-FeedForwardLayer_1_2/Gelu/mul:z:0Hmodel_9/transformer_encoder_28/Encoder-FeedForwardLayer_1_2/Gelu/add:z:0*
T0*+
_output_shapes
:���������P	�
Tmodel_9/transformer_encoder_28/Encoder-FeedForwardLayer_2_2/Tensordot/ReadVariableOpReadVariableOp]model_9_transformer_encoder_28_encoder_feedforwardlayer_2_2_tensordot_readvariableop_resource*
_output_shapes

:		*
dtype0�
Jmodel_9/transformer_encoder_28/Encoder-FeedForwardLayer_2_2/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:�
Jmodel_9/transformer_encoder_28/Encoder-FeedForwardLayer_2_2/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       �
Kmodel_9/transformer_encoder_28/Encoder-FeedForwardLayer_2_2/Tensordot/ShapeShapeJmodel_9/transformer_encoder_28/Encoder-FeedForwardLayer_1_2/Gelu/mul_1:z:0*
T0*
_output_shapes
::���
Smodel_9/transformer_encoder_28/Encoder-FeedForwardLayer_2_2/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Nmodel_9/transformer_encoder_28/Encoder-FeedForwardLayer_2_2/Tensordot/GatherV2GatherV2Tmodel_9/transformer_encoder_28/Encoder-FeedForwardLayer_2_2/Tensordot/Shape:output:0Smodel_9/transformer_encoder_28/Encoder-FeedForwardLayer_2_2/Tensordot/free:output:0\model_9/transformer_encoder_28/Encoder-FeedForwardLayer_2_2/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
Umodel_9/transformer_encoder_28/Encoder-FeedForwardLayer_2_2/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Pmodel_9/transformer_encoder_28/Encoder-FeedForwardLayer_2_2/Tensordot/GatherV2_1GatherV2Tmodel_9/transformer_encoder_28/Encoder-FeedForwardLayer_2_2/Tensordot/Shape:output:0Smodel_9/transformer_encoder_28/Encoder-FeedForwardLayer_2_2/Tensordot/axes:output:0^model_9/transformer_encoder_28/Encoder-FeedForwardLayer_2_2/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
Kmodel_9/transformer_encoder_28/Encoder-FeedForwardLayer_2_2/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
Jmodel_9/transformer_encoder_28/Encoder-FeedForwardLayer_2_2/Tensordot/ProdProdWmodel_9/transformer_encoder_28/Encoder-FeedForwardLayer_2_2/Tensordot/GatherV2:output:0Tmodel_9/transformer_encoder_28/Encoder-FeedForwardLayer_2_2/Tensordot/Const:output:0*
T0*
_output_shapes
: �
Mmodel_9/transformer_encoder_28/Encoder-FeedForwardLayer_2_2/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
Lmodel_9/transformer_encoder_28/Encoder-FeedForwardLayer_2_2/Tensordot/Prod_1ProdYmodel_9/transformer_encoder_28/Encoder-FeedForwardLayer_2_2/Tensordot/GatherV2_1:output:0Vmodel_9/transformer_encoder_28/Encoder-FeedForwardLayer_2_2/Tensordot/Const_1:output:0*
T0*
_output_shapes
: �
Qmodel_9/transformer_encoder_28/Encoder-FeedForwardLayer_2_2/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Lmodel_9/transformer_encoder_28/Encoder-FeedForwardLayer_2_2/Tensordot/concatConcatV2Smodel_9/transformer_encoder_28/Encoder-FeedForwardLayer_2_2/Tensordot/free:output:0Smodel_9/transformer_encoder_28/Encoder-FeedForwardLayer_2_2/Tensordot/axes:output:0Zmodel_9/transformer_encoder_28/Encoder-FeedForwardLayer_2_2/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
Kmodel_9/transformer_encoder_28/Encoder-FeedForwardLayer_2_2/Tensordot/stackPackSmodel_9/transformer_encoder_28/Encoder-FeedForwardLayer_2_2/Tensordot/Prod:output:0Umodel_9/transformer_encoder_28/Encoder-FeedForwardLayer_2_2/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
Omodel_9/transformer_encoder_28/Encoder-FeedForwardLayer_2_2/Tensordot/transpose	TransposeJmodel_9/transformer_encoder_28/Encoder-FeedForwardLayer_1_2/Gelu/mul_1:z:0Umodel_9/transformer_encoder_28/Encoder-FeedForwardLayer_2_2/Tensordot/concat:output:0*
T0*+
_output_shapes
:���������P	�
Mmodel_9/transformer_encoder_28/Encoder-FeedForwardLayer_2_2/Tensordot/ReshapeReshapeSmodel_9/transformer_encoder_28/Encoder-FeedForwardLayer_2_2/Tensordot/transpose:y:0Tmodel_9/transformer_encoder_28/Encoder-FeedForwardLayer_2_2/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
Lmodel_9/transformer_encoder_28/Encoder-FeedForwardLayer_2_2/Tensordot/MatMulMatMulVmodel_9/transformer_encoder_28/Encoder-FeedForwardLayer_2_2/Tensordot/Reshape:output:0\model_9/transformer_encoder_28/Encoder-FeedForwardLayer_2_2/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������	�
Mmodel_9/transformer_encoder_28/Encoder-FeedForwardLayer_2_2/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:	�
Smodel_9/transformer_encoder_28/Encoder-FeedForwardLayer_2_2/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Nmodel_9/transformer_encoder_28/Encoder-FeedForwardLayer_2_2/Tensordot/concat_1ConcatV2Wmodel_9/transformer_encoder_28/Encoder-FeedForwardLayer_2_2/Tensordot/GatherV2:output:0Vmodel_9/transformer_encoder_28/Encoder-FeedForwardLayer_2_2/Tensordot/Const_2:output:0\model_9/transformer_encoder_28/Encoder-FeedForwardLayer_2_2/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
Emodel_9/transformer_encoder_28/Encoder-FeedForwardLayer_2_2/TensordotReshapeVmodel_9/transformer_encoder_28/Encoder-FeedForwardLayer_2_2/Tensordot/MatMul:product:0Wmodel_9/transformer_encoder_28/Encoder-FeedForwardLayer_2_2/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������P	�
Rmodel_9/transformer_encoder_28/Encoder-FeedForwardLayer_2_2/BiasAdd/ReadVariableOpReadVariableOp[model_9_transformer_encoder_28_encoder_feedforwardlayer_2_2_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype0�
Cmodel_9/transformer_encoder_28/Encoder-FeedForwardLayer_2_2/BiasAddBiasAddNmodel_9/transformer_encoder_28/Encoder-FeedForwardLayer_2_2/Tensordot:output:0Zmodel_9/transformer_encoder_28/Encoder-FeedForwardLayer_2_2/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������P	�
2model_9/transformer_encoder_28/dropout_28/IdentityIdentityLmodel_9/transformer_encoder_28/Encoder-FeedForwardLayer_2_2/BiasAdd:output:0*
T0*+
_output_shapes
:���������P	�
>model_9/transformer_encoder_28/Encoder-2nd-AdditionLayer-2/addAddV2Bmodel_9/transformer_encoder_28/Encoder-1st-AdditionLayer-2/add:z:0;model_9/transformer_encoder_28/dropout_28/Identity:output:0*
T0*+
_output_shapes
:���������P	�
Emodel_9/transformer_encoder_29/Encoder-1st-NormalizationLayer-3/ShapeShapeBmodel_9/transformer_encoder_28/Encoder-2nd-AdditionLayer-2/add:z:0*
T0*
_output_shapes
::���
Smodel_9/transformer_encoder_29/Encoder-1st-NormalizationLayer-3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
Umodel_9/transformer_encoder_29/Encoder-1st-NormalizationLayer-3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
Umodel_9/transformer_encoder_29/Encoder-1st-NormalizationLayer-3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
Mmodel_9/transformer_encoder_29/Encoder-1st-NormalizationLayer-3/strided_sliceStridedSliceNmodel_9/transformer_encoder_29/Encoder-1st-NormalizationLayer-3/Shape:output:0\model_9/transformer_encoder_29/Encoder-1st-NormalizationLayer-3/strided_slice/stack:output:0^model_9/transformer_encoder_29/Encoder-1st-NormalizationLayer-3/strided_slice/stack_1:output:0^model_9/transformer_encoder_29/Encoder-1st-NormalizationLayer-3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask�
Emodel_9/transformer_encoder_29/Encoder-1st-NormalizationLayer-3/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
Dmodel_9/transformer_encoder_29/Encoder-1st-NormalizationLayer-3/ProdProdVmodel_9/transformer_encoder_29/Encoder-1st-NormalizationLayer-3/strided_slice:output:0Nmodel_9/transformer_encoder_29/Encoder-1st-NormalizationLayer-3/Const:output:0*
T0*
_output_shapes
: �
Umodel_9/transformer_encoder_29/Encoder-1st-NormalizationLayer-3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:�
Wmodel_9/transformer_encoder_29/Encoder-1st-NormalizationLayer-3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: �
Wmodel_9/transformer_encoder_29/Encoder-1st-NormalizationLayer-3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
Omodel_9/transformer_encoder_29/Encoder-1st-NormalizationLayer-3/strided_slice_1StridedSliceNmodel_9/transformer_encoder_29/Encoder-1st-NormalizationLayer-3/Shape:output:0^model_9/transformer_encoder_29/Encoder-1st-NormalizationLayer-3/strided_slice_1/stack:output:0`model_9/transformer_encoder_29/Encoder-1st-NormalizationLayer-3/strided_slice_1/stack_1:output:0`model_9/transformer_encoder_29/Encoder-1st-NormalizationLayer-3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask�
Gmodel_9/transformer_encoder_29/Encoder-1st-NormalizationLayer-3/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
Fmodel_9/transformer_encoder_29/Encoder-1st-NormalizationLayer-3/Prod_1ProdXmodel_9/transformer_encoder_29/Encoder-1st-NormalizationLayer-3/strided_slice_1:output:0Pmodel_9/transformer_encoder_29/Encoder-1st-NormalizationLayer-3/Const_1:output:0*
T0*
_output_shapes
: �
Omodel_9/transformer_encoder_29/Encoder-1st-NormalizationLayer-3/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :�
Omodel_9/transformer_encoder_29/Encoder-1st-NormalizationLayer-3/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :�
Mmodel_9/transformer_encoder_29/Encoder-1st-NormalizationLayer-3/Reshape/shapePackXmodel_9/transformer_encoder_29/Encoder-1st-NormalizationLayer-3/Reshape/shape/0:output:0Mmodel_9/transformer_encoder_29/Encoder-1st-NormalizationLayer-3/Prod:output:0Omodel_9/transformer_encoder_29/Encoder-1st-NormalizationLayer-3/Prod_1:output:0Xmodel_9/transformer_encoder_29/Encoder-1st-NormalizationLayer-3/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:�
Gmodel_9/transformer_encoder_29/Encoder-1st-NormalizationLayer-3/ReshapeReshapeBmodel_9/transformer_encoder_28/Encoder-2nd-AdditionLayer-2/add:z:0Vmodel_9/transformer_encoder_29/Encoder-1st-NormalizationLayer-3/Reshape/shape:output:0*
T0*8
_output_shapes&
$:"�������������������
Kmodel_9/transformer_encoder_29/Encoder-1st-NormalizationLayer-3/ones/packedPackMmodel_9/transformer_encoder_29/Encoder-1st-NormalizationLayer-3/Prod:output:0*
N*
T0*
_output_shapes
:�
Jmodel_9/transformer_encoder_29/Encoder-1st-NormalizationLayer-3/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
Dmodel_9/transformer_encoder_29/Encoder-1st-NormalizationLayer-3/onesFillTmodel_9/transformer_encoder_29/Encoder-1st-NormalizationLayer-3/ones/packed:output:0Smodel_9/transformer_encoder_29/Encoder-1st-NormalizationLayer-3/ones/Const:output:0*
T0*#
_output_shapes
:����������
Lmodel_9/transformer_encoder_29/Encoder-1st-NormalizationLayer-3/zeros/packedPackMmodel_9/transformer_encoder_29/Encoder-1st-NormalizationLayer-3/Prod:output:0*
N*
T0*
_output_shapes
:�
Kmodel_9/transformer_encoder_29/Encoder-1st-NormalizationLayer-3/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
Emodel_9/transformer_encoder_29/Encoder-1st-NormalizationLayer-3/zerosFillUmodel_9/transformer_encoder_29/Encoder-1st-NormalizationLayer-3/zeros/packed:output:0Tmodel_9/transformer_encoder_29/Encoder-1st-NormalizationLayer-3/zeros/Const:output:0*
T0*#
_output_shapes
:����������
Gmodel_9/transformer_encoder_29/Encoder-1st-NormalizationLayer-3/Const_2Const*
_output_shapes
: *
dtype0*
valueB �
Gmodel_9/transformer_encoder_29/Encoder-1st-NormalizationLayer-3/Const_3Const*
_output_shapes
: *
dtype0*
valueB �
Pmodel_9/transformer_encoder_29/Encoder-1st-NormalizationLayer-3/FusedBatchNormV3FusedBatchNormV3Pmodel_9/transformer_encoder_29/Encoder-1st-NormalizationLayer-3/Reshape:output:0Mmodel_9/transformer_encoder_29/Encoder-1st-NormalizationLayer-3/ones:output:0Nmodel_9/transformer_encoder_29/Encoder-1st-NormalizationLayer-3/zeros:output:0Pmodel_9/transformer_encoder_29/Encoder-1st-NormalizationLayer-3/Const_2:output:0Pmodel_9/transformer_encoder_29/Encoder-1st-NormalizationLayer-3/Const_3:output:0*
T0*
U0*x
_output_shapesf
d:"������������������:���������:���������:���������:���������:*
data_formatNCHW*
epsilon%o�:�
Imodel_9/transformer_encoder_29/Encoder-1st-NormalizationLayer-3/Reshape_1ReshapeTmodel_9/transformer_encoder_29/Encoder-1st-NormalizationLayer-3/FusedBatchNormV3:y:0Nmodel_9/transformer_encoder_29/Encoder-1st-NormalizationLayer-3/Shape:output:0*
T0*+
_output_shapes
:���������P	�
Rmodel_9/transformer_encoder_29/Encoder-1st-NormalizationLayer-3/mul/ReadVariableOpReadVariableOp[model_9_transformer_encoder_29_encoder_1st_normalizationlayer_3_mul_readvariableop_resource*
_output_shapes
:	*
dtype0�
Cmodel_9/transformer_encoder_29/Encoder-1st-NormalizationLayer-3/mulMulRmodel_9/transformer_encoder_29/Encoder-1st-NormalizationLayer-3/Reshape_1:output:0Zmodel_9/transformer_encoder_29/Encoder-1st-NormalizationLayer-3/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������P	�
Rmodel_9/transformer_encoder_29/Encoder-1st-NormalizationLayer-3/add/ReadVariableOpReadVariableOp[model_9_transformer_encoder_29_encoder_1st_normalizationlayer_3_add_readvariableop_resource*
_output_shapes
:	*
dtype0�
Cmodel_9/transformer_encoder_29/Encoder-1st-NormalizationLayer-3/addAddV2Gmodel_9/transformer_encoder_29/Encoder-1st-NormalizationLayer-3/mul:z:0Zmodel_9/transformer_encoder_29/Encoder-1st-NormalizationLayer-3/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������P	�
^model_9/transformer_encoder_29/Encoder-SelfAttentionLayer-3/query/einsum/Einsum/ReadVariableOpReadVariableOpgmodel_9_transformer_encoder_29_encoder_selfattentionlayer_3_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:	*
dtype0�
Omodel_9/transformer_encoder_29/Encoder-SelfAttentionLayer-3/query/einsum/EinsumEinsumGmodel_9/transformer_encoder_29/Encoder-1st-NormalizationLayer-3/add:z:0fmodel_9/transformer_encoder_29/Encoder-SelfAttentionLayer-3/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������P*
equationabc,cde->abde�
Tmodel_9/transformer_encoder_29/Encoder-SelfAttentionLayer-3/query/add/ReadVariableOpReadVariableOp]model_9_transformer_encoder_29_encoder_selfattentionlayer_3_query_add_readvariableop_resource*
_output_shapes

:*
dtype0�
Emodel_9/transformer_encoder_29/Encoder-SelfAttentionLayer-3/query/addAddV2Xmodel_9/transformer_encoder_29/Encoder-SelfAttentionLayer-3/query/einsum/Einsum:output:0\model_9/transformer_encoder_29/Encoder-SelfAttentionLayer-3/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������P�
\model_9/transformer_encoder_29/Encoder-SelfAttentionLayer-3/key/einsum/Einsum/ReadVariableOpReadVariableOpemodel_9_transformer_encoder_29_encoder_selfattentionlayer_3_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:	*
dtype0�
Mmodel_9/transformer_encoder_29/Encoder-SelfAttentionLayer-3/key/einsum/EinsumEinsumGmodel_9/transformer_encoder_29/Encoder-1st-NormalizationLayer-3/add:z:0dmodel_9/transformer_encoder_29/Encoder-SelfAttentionLayer-3/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������P*
equationabc,cde->abde�
Rmodel_9/transformer_encoder_29/Encoder-SelfAttentionLayer-3/key/add/ReadVariableOpReadVariableOp[model_9_transformer_encoder_29_encoder_selfattentionlayer_3_key_add_readvariableop_resource*
_output_shapes

:*
dtype0�
Cmodel_9/transformer_encoder_29/Encoder-SelfAttentionLayer-3/key/addAddV2Vmodel_9/transformer_encoder_29/Encoder-SelfAttentionLayer-3/key/einsum/Einsum:output:0Zmodel_9/transformer_encoder_29/Encoder-SelfAttentionLayer-3/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������P�
^model_9/transformer_encoder_29/Encoder-SelfAttentionLayer-3/value/einsum/Einsum/ReadVariableOpReadVariableOpgmodel_9_transformer_encoder_29_encoder_selfattentionlayer_3_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:	*
dtype0�
Omodel_9/transformer_encoder_29/Encoder-SelfAttentionLayer-3/value/einsum/EinsumEinsumGmodel_9/transformer_encoder_29/Encoder-1st-NormalizationLayer-3/add:z:0fmodel_9/transformer_encoder_29/Encoder-SelfAttentionLayer-3/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������P*
equationabc,cde->abde�
Tmodel_9/transformer_encoder_29/Encoder-SelfAttentionLayer-3/value/add/ReadVariableOpReadVariableOp]model_9_transformer_encoder_29_encoder_selfattentionlayer_3_value_add_readvariableop_resource*
_output_shapes

:*
dtype0�
Emodel_9/transformer_encoder_29/Encoder-SelfAttentionLayer-3/value/addAddV2Xmodel_9/transformer_encoder_29/Encoder-SelfAttentionLayer-3/value/einsum/Einsum:output:0\model_9/transformer_encoder_29/Encoder-SelfAttentionLayer-3/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������P�
Amodel_9/transformer_encoder_29/Encoder-SelfAttentionLayer-3/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *:�?�
?model_9/transformer_encoder_29/Encoder-SelfAttentionLayer-3/MulMulImodel_9/transformer_encoder_29/Encoder-SelfAttentionLayer-3/query/add:z:0Jmodel_9/transformer_encoder_29/Encoder-SelfAttentionLayer-3/Mul/y:output:0*
T0*/
_output_shapes
:���������P�
Imodel_9/transformer_encoder_29/Encoder-SelfAttentionLayer-3/einsum/EinsumEinsumGmodel_9/transformer_encoder_29/Encoder-SelfAttentionLayer-3/key/add:z:0Cmodel_9/transformer_encoder_29/Encoder-SelfAttentionLayer-3/Mul:z:0*
N*
T0*/
_output_shapes
:���������PP*
equationaecd,abcd->acbe�
Kmodel_9/transformer_encoder_29/Encoder-SelfAttentionLayer-3/softmax/SoftmaxSoftmaxRmodel_9/transformer_encoder_29/Encoder-SelfAttentionLayer-3/einsum/Einsum:output:0*
T0*/
_output_shapes
:���������PP�
Lmodel_9/transformer_encoder_29/Encoder-SelfAttentionLayer-3/dropout/IdentityIdentityUmodel_9/transformer_encoder_29/Encoder-SelfAttentionLayer-3/softmax/Softmax:softmax:0*
T0*/
_output_shapes
:���������PP�
Kmodel_9/transformer_encoder_29/Encoder-SelfAttentionLayer-3/einsum_1/EinsumEinsumUmodel_9/transformer_encoder_29/Encoder-SelfAttentionLayer-3/dropout/Identity:output:0Imodel_9/transformer_encoder_29/Encoder-SelfAttentionLayer-3/value/add:z:0*
N*
T0*/
_output_shapes
:���������P*
equationacbe,aecd->abcd�
imodel_9/transformer_encoder_29/Encoder-SelfAttentionLayer-3/attention_output/einsum/Einsum/ReadVariableOpReadVariableOprmodel_9_transformer_encoder_29_encoder_selfattentionlayer_3_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:	*
dtype0�
Zmodel_9/transformer_encoder_29/Encoder-SelfAttentionLayer-3/attention_output/einsum/EinsumEinsumTmodel_9/transformer_encoder_29/Encoder-SelfAttentionLayer-3/einsum_1/Einsum:output:0qmodel_9/transformer_encoder_29/Encoder-SelfAttentionLayer-3/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:���������P	*
equationabcd,cde->abe�
_model_9/transformer_encoder_29/Encoder-SelfAttentionLayer-3/attention_output/add/ReadVariableOpReadVariableOphmodel_9_transformer_encoder_29_encoder_selfattentionlayer_3_attention_output_add_readvariableop_resource*
_output_shapes
:	*
dtype0�
Pmodel_9/transformer_encoder_29/Encoder-SelfAttentionLayer-3/attention_output/addAddV2cmodel_9/transformer_encoder_29/Encoder-SelfAttentionLayer-3/attention_output/einsum/Einsum:output:0gmodel_9/transformer_encoder_29/Encoder-SelfAttentionLayer-3/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������P	�
>model_9/transformer_encoder_29/Encoder-1st-AdditionLayer-3/addAddV2Bmodel_9/transformer_encoder_28/Encoder-2nd-AdditionLayer-2/add:z:0Tmodel_9/transformer_encoder_29/Encoder-SelfAttentionLayer-3/attention_output/add:z:0*
T0*+
_output_shapes
:���������P	�
Emodel_9/transformer_encoder_29/Encoder-2nd-NormalizationLayer-3/ShapeShapeBmodel_9/transformer_encoder_29/Encoder-1st-AdditionLayer-3/add:z:0*
T0*
_output_shapes
::���
Smodel_9/transformer_encoder_29/Encoder-2nd-NormalizationLayer-3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
Umodel_9/transformer_encoder_29/Encoder-2nd-NormalizationLayer-3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
Umodel_9/transformer_encoder_29/Encoder-2nd-NormalizationLayer-3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
Mmodel_9/transformer_encoder_29/Encoder-2nd-NormalizationLayer-3/strided_sliceStridedSliceNmodel_9/transformer_encoder_29/Encoder-2nd-NormalizationLayer-3/Shape:output:0\model_9/transformer_encoder_29/Encoder-2nd-NormalizationLayer-3/strided_slice/stack:output:0^model_9/transformer_encoder_29/Encoder-2nd-NormalizationLayer-3/strided_slice/stack_1:output:0^model_9/transformer_encoder_29/Encoder-2nd-NormalizationLayer-3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask�
Emodel_9/transformer_encoder_29/Encoder-2nd-NormalizationLayer-3/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
Dmodel_9/transformer_encoder_29/Encoder-2nd-NormalizationLayer-3/ProdProdVmodel_9/transformer_encoder_29/Encoder-2nd-NormalizationLayer-3/strided_slice:output:0Nmodel_9/transformer_encoder_29/Encoder-2nd-NormalizationLayer-3/Const:output:0*
T0*
_output_shapes
: �
Umodel_9/transformer_encoder_29/Encoder-2nd-NormalizationLayer-3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:�
Wmodel_9/transformer_encoder_29/Encoder-2nd-NormalizationLayer-3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: �
Wmodel_9/transformer_encoder_29/Encoder-2nd-NormalizationLayer-3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
Omodel_9/transformer_encoder_29/Encoder-2nd-NormalizationLayer-3/strided_slice_1StridedSliceNmodel_9/transformer_encoder_29/Encoder-2nd-NormalizationLayer-3/Shape:output:0^model_9/transformer_encoder_29/Encoder-2nd-NormalizationLayer-3/strided_slice_1/stack:output:0`model_9/transformer_encoder_29/Encoder-2nd-NormalizationLayer-3/strided_slice_1/stack_1:output:0`model_9/transformer_encoder_29/Encoder-2nd-NormalizationLayer-3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask�
Gmodel_9/transformer_encoder_29/Encoder-2nd-NormalizationLayer-3/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
Fmodel_9/transformer_encoder_29/Encoder-2nd-NormalizationLayer-3/Prod_1ProdXmodel_9/transformer_encoder_29/Encoder-2nd-NormalizationLayer-3/strided_slice_1:output:0Pmodel_9/transformer_encoder_29/Encoder-2nd-NormalizationLayer-3/Const_1:output:0*
T0*
_output_shapes
: �
Omodel_9/transformer_encoder_29/Encoder-2nd-NormalizationLayer-3/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :�
Omodel_9/transformer_encoder_29/Encoder-2nd-NormalizationLayer-3/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :�
Mmodel_9/transformer_encoder_29/Encoder-2nd-NormalizationLayer-3/Reshape/shapePackXmodel_9/transformer_encoder_29/Encoder-2nd-NormalizationLayer-3/Reshape/shape/0:output:0Mmodel_9/transformer_encoder_29/Encoder-2nd-NormalizationLayer-3/Prod:output:0Omodel_9/transformer_encoder_29/Encoder-2nd-NormalizationLayer-3/Prod_1:output:0Xmodel_9/transformer_encoder_29/Encoder-2nd-NormalizationLayer-3/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:�
Gmodel_9/transformer_encoder_29/Encoder-2nd-NormalizationLayer-3/ReshapeReshapeBmodel_9/transformer_encoder_29/Encoder-1st-AdditionLayer-3/add:z:0Vmodel_9/transformer_encoder_29/Encoder-2nd-NormalizationLayer-3/Reshape/shape:output:0*
T0*8
_output_shapes&
$:"�������������������
Kmodel_9/transformer_encoder_29/Encoder-2nd-NormalizationLayer-3/ones/packedPackMmodel_9/transformer_encoder_29/Encoder-2nd-NormalizationLayer-3/Prod:output:0*
N*
T0*
_output_shapes
:�
Jmodel_9/transformer_encoder_29/Encoder-2nd-NormalizationLayer-3/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
Dmodel_9/transformer_encoder_29/Encoder-2nd-NormalizationLayer-3/onesFillTmodel_9/transformer_encoder_29/Encoder-2nd-NormalizationLayer-3/ones/packed:output:0Smodel_9/transformer_encoder_29/Encoder-2nd-NormalizationLayer-3/ones/Const:output:0*
T0*#
_output_shapes
:����������
Lmodel_9/transformer_encoder_29/Encoder-2nd-NormalizationLayer-3/zeros/packedPackMmodel_9/transformer_encoder_29/Encoder-2nd-NormalizationLayer-3/Prod:output:0*
N*
T0*
_output_shapes
:�
Kmodel_9/transformer_encoder_29/Encoder-2nd-NormalizationLayer-3/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
Emodel_9/transformer_encoder_29/Encoder-2nd-NormalizationLayer-3/zerosFillUmodel_9/transformer_encoder_29/Encoder-2nd-NormalizationLayer-3/zeros/packed:output:0Tmodel_9/transformer_encoder_29/Encoder-2nd-NormalizationLayer-3/zeros/Const:output:0*
T0*#
_output_shapes
:����������
Gmodel_9/transformer_encoder_29/Encoder-2nd-NormalizationLayer-3/Const_2Const*
_output_shapes
: *
dtype0*
valueB �
Gmodel_9/transformer_encoder_29/Encoder-2nd-NormalizationLayer-3/Const_3Const*
_output_shapes
: *
dtype0*
valueB �
Pmodel_9/transformer_encoder_29/Encoder-2nd-NormalizationLayer-3/FusedBatchNormV3FusedBatchNormV3Pmodel_9/transformer_encoder_29/Encoder-2nd-NormalizationLayer-3/Reshape:output:0Mmodel_9/transformer_encoder_29/Encoder-2nd-NormalizationLayer-3/ones:output:0Nmodel_9/transformer_encoder_29/Encoder-2nd-NormalizationLayer-3/zeros:output:0Pmodel_9/transformer_encoder_29/Encoder-2nd-NormalizationLayer-3/Const_2:output:0Pmodel_9/transformer_encoder_29/Encoder-2nd-NormalizationLayer-3/Const_3:output:0*
T0*
U0*x
_output_shapesf
d:"������������������:���������:���������:���������:���������:*
data_formatNCHW*
epsilon%o�:�
Imodel_9/transformer_encoder_29/Encoder-2nd-NormalizationLayer-3/Reshape_1ReshapeTmodel_9/transformer_encoder_29/Encoder-2nd-NormalizationLayer-3/FusedBatchNormV3:y:0Nmodel_9/transformer_encoder_29/Encoder-2nd-NormalizationLayer-3/Shape:output:0*
T0*+
_output_shapes
:���������P	�
Rmodel_9/transformer_encoder_29/Encoder-2nd-NormalizationLayer-3/mul/ReadVariableOpReadVariableOp[model_9_transformer_encoder_29_encoder_2nd_normalizationlayer_3_mul_readvariableop_resource*
_output_shapes
:	*
dtype0�
Cmodel_9/transformer_encoder_29/Encoder-2nd-NormalizationLayer-3/mulMulRmodel_9/transformer_encoder_29/Encoder-2nd-NormalizationLayer-3/Reshape_1:output:0Zmodel_9/transformer_encoder_29/Encoder-2nd-NormalizationLayer-3/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������P	�
Rmodel_9/transformer_encoder_29/Encoder-2nd-NormalizationLayer-3/add/ReadVariableOpReadVariableOp[model_9_transformer_encoder_29_encoder_2nd_normalizationlayer_3_add_readvariableop_resource*
_output_shapes
:	*
dtype0�
Cmodel_9/transformer_encoder_29/Encoder-2nd-NormalizationLayer-3/addAddV2Gmodel_9/transformer_encoder_29/Encoder-2nd-NormalizationLayer-3/mul:z:0Zmodel_9/transformer_encoder_29/Encoder-2nd-NormalizationLayer-3/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������P	�
Tmodel_9/transformer_encoder_29/Encoder-FeedForwardLayer_1_3/Tensordot/ReadVariableOpReadVariableOp]model_9_transformer_encoder_29_encoder_feedforwardlayer_1_3_tensordot_readvariableop_resource*
_output_shapes

:		*
dtype0�
Jmodel_9/transformer_encoder_29/Encoder-FeedForwardLayer_1_3/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:�
Jmodel_9/transformer_encoder_29/Encoder-FeedForwardLayer_1_3/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       �
Kmodel_9/transformer_encoder_29/Encoder-FeedForwardLayer_1_3/Tensordot/ShapeShapeGmodel_9/transformer_encoder_29/Encoder-2nd-NormalizationLayer-3/add:z:0*
T0*
_output_shapes
::���
Smodel_9/transformer_encoder_29/Encoder-FeedForwardLayer_1_3/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Nmodel_9/transformer_encoder_29/Encoder-FeedForwardLayer_1_3/Tensordot/GatherV2GatherV2Tmodel_9/transformer_encoder_29/Encoder-FeedForwardLayer_1_3/Tensordot/Shape:output:0Smodel_9/transformer_encoder_29/Encoder-FeedForwardLayer_1_3/Tensordot/free:output:0\model_9/transformer_encoder_29/Encoder-FeedForwardLayer_1_3/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
Umodel_9/transformer_encoder_29/Encoder-FeedForwardLayer_1_3/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Pmodel_9/transformer_encoder_29/Encoder-FeedForwardLayer_1_3/Tensordot/GatherV2_1GatherV2Tmodel_9/transformer_encoder_29/Encoder-FeedForwardLayer_1_3/Tensordot/Shape:output:0Smodel_9/transformer_encoder_29/Encoder-FeedForwardLayer_1_3/Tensordot/axes:output:0^model_9/transformer_encoder_29/Encoder-FeedForwardLayer_1_3/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
Kmodel_9/transformer_encoder_29/Encoder-FeedForwardLayer_1_3/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
Jmodel_9/transformer_encoder_29/Encoder-FeedForwardLayer_1_3/Tensordot/ProdProdWmodel_9/transformer_encoder_29/Encoder-FeedForwardLayer_1_3/Tensordot/GatherV2:output:0Tmodel_9/transformer_encoder_29/Encoder-FeedForwardLayer_1_3/Tensordot/Const:output:0*
T0*
_output_shapes
: �
Mmodel_9/transformer_encoder_29/Encoder-FeedForwardLayer_1_3/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
Lmodel_9/transformer_encoder_29/Encoder-FeedForwardLayer_1_3/Tensordot/Prod_1ProdYmodel_9/transformer_encoder_29/Encoder-FeedForwardLayer_1_3/Tensordot/GatherV2_1:output:0Vmodel_9/transformer_encoder_29/Encoder-FeedForwardLayer_1_3/Tensordot/Const_1:output:0*
T0*
_output_shapes
: �
Qmodel_9/transformer_encoder_29/Encoder-FeedForwardLayer_1_3/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Lmodel_9/transformer_encoder_29/Encoder-FeedForwardLayer_1_3/Tensordot/concatConcatV2Smodel_9/transformer_encoder_29/Encoder-FeedForwardLayer_1_3/Tensordot/free:output:0Smodel_9/transformer_encoder_29/Encoder-FeedForwardLayer_1_3/Tensordot/axes:output:0Zmodel_9/transformer_encoder_29/Encoder-FeedForwardLayer_1_3/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
Kmodel_9/transformer_encoder_29/Encoder-FeedForwardLayer_1_3/Tensordot/stackPackSmodel_9/transformer_encoder_29/Encoder-FeedForwardLayer_1_3/Tensordot/Prod:output:0Umodel_9/transformer_encoder_29/Encoder-FeedForwardLayer_1_3/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
Omodel_9/transformer_encoder_29/Encoder-FeedForwardLayer_1_3/Tensordot/transpose	TransposeGmodel_9/transformer_encoder_29/Encoder-2nd-NormalizationLayer-3/add:z:0Umodel_9/transformer_encoder_29/Encoder-FeedForwardLayer_1_3/Tensordot/concat:output:0*
T0*+
_output_shapes
:���������P	�
Mmodel_9/transformer_encoder_29/Encoder-FeedForwardLayer_1_3/Tensordot/ReshapeReshapeSmodel_9/transformer_encoder_29/Encoder-FeedForwardLayer_1_3/Tensordot/transpose:y:0Tmodel_9/transformer_encoder_29/Encoder-FeedForwardLayer_1_3/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
Lmodel_9/transformer_encoder_29/Encoder-FeedForwardLayer_1_3/Tensordot/MatMulMatMulVmodel_9/transformer_encoder_29/Encoder-FeedForwardLayer_1_3/Tensordot/Reshape:output:0\model_9/transformer_encoder_29/Encoder-FeedForwardLayer_1_3/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������	�
Mmodel_9/transformer_encoder_29/Encoder-FeedForwardLayer_1_3/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:	�
Smodel_9/transformer_encoder_29/Encoder-FeedForwardLayer_1_3/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Nmodel_9/transformer_encoder_29/Encoder-FeedForwardLayer_1_3/Tensordot/concat_1ConcatV2Wmodel_9/transformer_encoder_29/Encoder-FeedForwardLayer_1_3/Tensordot/GatherV2:output:0Vmodel_9/transformer_encoder_29/Encoder-FeedForwardLayer_1_3/Tensordot/Const_2:output:0\model_9/transformer_encoder_29/Encoder-FeedForwardLayer_1_3/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
Emodel_9/transformer_encoder_29/Encoder-FeedForwardLayer_1_3/TensordotReshapeVmodel_9/transformer_encoder_29/Encoder-FeedForwardLayer_1_3/Tensordot/MatMul:product:0Wmodel_9/transformer_encoder_29/Encoder-FeedForwardLayer_1_3/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������P	�
Rmodel_9/transformer_encoder_29/Encoder-FeedForwardLayer_1_3/BiasAdd/ReadVariableOpReadVariableOp[model_9_transformer_encoder_29_encoder_feedforwardlayer_1_3_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype0�
Cmodel_9/transformer_encoder_29/Encoder-FeedForwardLayer_1_3/BiasAddBiasAddNmodel_9/transformer_encoder_29/Encoder-FeedForwardLayer_1_3/Tensordot:output:0Zmodel_9/transformer_encoder_29/Encoder-FeedForwardLayer_1_3/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������P	�
Fmodel_9/transformer_encoder_29/Encoder-FeedForwardLayer_1_3/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
Dmodel_9/transformer_encoder_29/Encoder-FeedForwardLayer_1_3/Gelu/mulMulOmodel_9/transformer_encoder_29/Encoder-FeedForwardLayer_1_3/Gelu/mul/x:output:0Lmodel_9/transformer_encoder_29/Encoder-FeedForwardLayer_1_3/BiasAdd:output:0*
T0*+
_output_shapes
:���������P	�
Gmodel_9/transformer_encoder_29/Encoder-FeedForwardLayer_1_3/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *��?�
Hmodel_9/transformer_encoder_29/Encoder-FeedForwardLayer_1_3/Gelu/truedivRealDivLmodel_9/transformer_encoder_29/Encoder-FeedForwardLayer_1_3/BiasAdd:output:0Pmodel_9/transformer_encoder_29/Encoder-FeedForwardLayer_1_3/Gelu/Cast/x:output:0*
T0*+
_output_shapes
:���������P	�
Dmodel_9/transformer_encoder_29/Encoder-FeedForwardLayer_1_3/Gelu/ErfErfLmodel_9/transformer_encoder_29/Encoder-FeedForwardLayer_1_3/Gelu/truediv:z:0*
T0*+
_output_shapes
:���������P	�
Fmodel_9/transformer_encoder_29/Encoder-FeedForwardLayer_1_3/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
Dmodel_9/transformer_encoder_29/Encoder-FeedForwardLayer_1_3/Gelu/addAddV2Omodel_9/transformer_encoder_29/Encoder-FeedForwardLayer_1_3/Gelu/add/x:output:0Hmodel_9/transformer_encoder_29/Encoder-FeedForwardLayer_1_3/Gelu/Erf:y:0*
T0*+
_output_shapes
:���������P	�
Fmodel_9/transformer_encoder_29/Encoder-FeedForwardLayer_1_3/Gelu/mul_1MulHmodel_9/transformer_encoder_29/Encoder-FeedForwardLayer_1_3/Gelu/mul:z:0Hmodel_9/transformer_encoder_29/Encoder-FeedForwardLayer_1_3/Gelu/add:z:0*
T0*+
_output_shapes
:���������P	�
Tmodel_9/transformer_encoder_29/Encoder-FeedForwardLayer_2_3/Tensordot/ReadVariableOpReadVariableOp]model_9_transformer_encoder_29_encoder_feedforwardlayer_2_3_tensordot_readvariableop_resource*
_output_shapes

:		*
dtype0�
Jmodel_9/transformer_encoder_29/Encoder-FeedForwardLayer_2_3/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:�
Jmodel_9/transformer_encoder_29/Encoder-FeedForwardLayer_2_3/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       �
Kmodel_9/transformer_encoder_29/Encoder-FeedForwardLayer_2_3/Tensordot/ShapeShapeJmodel_9/transformer_encoder_29/Encoder-FeedForwardLayer_1_3/Gelu/mul_1:z:0*
T0*
_output_shapes
::���
Smodel_9/transformer_encoder_29/Encoder-FeedForwardLayer_2_3/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Nmodel_9/transformer_encoder_29/Encoder-FeedForwardLayer_2_3/Tensordot/GatherV2GatherV2Tmodel_9/transformer_encoder_29/Encoder-FeedForwardLayer_2_3/Tensordot/Shape:output:0Smodel_9/transformer_encoder_29/Encoder-FeedForwardLayer_2_3/Tensordot/free:output:0\model_9/transformer_encoder_29/Encoder-FeedForwardLayer_2_3/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
Umodel_9/transformer_encoder_29/Encoder-FeedForwardLayer_2_3/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Pmodel_9/transformer_encoder_29/Encoder-FeedForwardLayer_2_3/Tensordot/GatherV2_1GatherV2Tmodel_9/transformer_encoder_29/Encoder-FeedForwardLayer_2_3/Tensordot/Shape:output:0Smodel_9/transformer_encoder_29/Encoder-FeedForwardLayer_2_3/Tensordot/axes:output:0^model_9/transformer_encoder_29/Encoder-FeedForwardLayer_2_3/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
Kmodel_9/transformer_encoder_29/Encoder-FeedForwardLayer_2_3/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
Jmodel_9/transformer_encoder_29/Encoder-FeedForwardLayer_2_3/Tensordot/ProdProdWmodel_9/transformer_encoder_29/Encoder-FeedForwardLayer_2_3/Tensordot/GatherV2:output:0Tmodel_9/transformer_encoder_29/Encoder-FeedForwardLayer_2_3/Tensordot/Const:output:0*
T0*
_output_shapes
: �
Mmodel_9/transformer_encoder_29/Encoder-FeedForwardLayer_2_3/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
Lmodel_9/transformer_encoder_29/Encoder-FeedForwardLayer_2_3/Tensordot/Prod_1ProdYmodel_9/transformer_encoder_29/Encoder-FeedForwardLayer_2_3/Tensordot/GatherV2_1:output:0Vmodel_9/transformer_encoder_29/Encoder-FeedForwardLayer_2_3/Tensordot/Const_1:output:0*
T0*
_output_shapes
: �
Qmodel_9/transformer_encoder_29/Encoder-FeedForwardLayer_2_3/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Lmodel_9/transformer_encoder_29/Encoder-FeedForwardLayer_2_3/Tensordot/concatConcatV2Smodel_9/transformer_encoder_29/Encoder-FeedForwardLayer_2_3/Tensordot/free:output:0Smodel_9/transformer_encoder_29/Encoder-FeedForwardLayer_2_3/Tensordot/axes:output:0Zmodel_9/transformer_encoder_29/Encoder-FeedForwardLayer_2_3/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
Kmodel_9/transformer_encoder_29/Encoder-FeedForwardLayer_2_3/Tensordot/stackPackSmodel_9/transformer_encoder_29/Encoder-FeedForwardLayer_2_3/Tensordot/Prod:output:0Umodel_9/transformer_encoder_29/Encoder-FeedForwardLayer_2_3/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
Omodel_9/transformer_encoder_29/Encoder-FeedForwardLayer_2_3/Tensordot/transpose	TransposeJmodel_9/transformer_encoder_29/Encoder-FeedForwardLayer_1_3/Gelu/mul_1:z:0Umodel_9/transformer_encoder_29/Encoder-FeedForwardLayer_2_3/Tensordot/concat:output:0*
T0*+
_output_shapes
:���������P	�
Mmodel_9/transformer_encoder_29/Encoder-FeedForwardLayer_2_3/Tensordot/ReshapeReshapeSmodel_9/transformer_encoder_29/Encoder-FeedForwardLayer_2_3/Tensordot/transpose:y:0Tmodel_9/transformer_encoder_29/Encoder-FeedForwardLayer_2_3/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
Lmodel_9/transformer_encoder_29/Encoder-FeedForwardLayer_2_3/Tensordot/MatMulMatMulVmodel_9/transformer_encoder_29/Encoder-FeedForwardLayer_2_3/Tensordot/Reshape:output:0\model_9/transformer_encoder_29/Encoder-FeedForwardLayer_2_3/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������	�
Mmodel_9/transformer_encoder_29/Encoder-FeedForwardLayer_2_3/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:	�
Smodel_9/transformer_encoder_29/Encoder-FeedForwardLayer_2_3/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Nmodel_9/transformer_encoder_29/Encoder-FeedForwardLayer_2_3/Tensordot/concat_1ConcatV2Wmodel_9/transformer_encoder_29/Encoder-FeedForwardLayer_2_3/Tensordot/GatherV2:output:0Vmodel_9/transformer_encoder_29/Encoder-FeedForwardLayer_2_3/Tensordot/Const_2:output:0\model_9/transformer_encoder_29/Encoder-FeedForwardLayer_2_3/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
Emodel_9/transformer_encoder_29/Encoder-FeedForwardLayer_2_3/TensordotReshapeVmodel_9/transformer_encoder_29/Encoder-FeedForwardLayer_2_3/Tensordot/MatMul:product:0Wmodel_9/transformer_encoder_29/Encoder-FeedForwardLayer_2_3/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������P	�
Rmodel_9/transformer_encoder_29/Encoder-FeedForwardLayer_2_3/BiasAdd/ReadVariableOpReadVariableOp[model_9_transformer_encoder_29_encoder_feedforwardlayer_2_3_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype0�
Cmodel_9/transformer_encoder_29/Encoder-FeedForwardLayer_2_3/BiasAddBiasAddNmodel_9/transformer_encoder_29/Encoder-FeedForwardLayer_2_3/Tensordot:output:0Zmodel_9/transformer_encoder_29/Encoder-FeedForwardLayer_2_3/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������P	�
2model_9/transformer_encoder_29/dropout_29/IdentityIdentityLmodel_9/transformer_encoder_29/Encoder-FeedForwardLayer_2_3/BiasAdd:output:0*
T0*+
_output_shapes
:���������P	�
>model_9/transformer_encoder_29/Encoder-2nd-AdditionLayer-3/addAddV2Bmodel_9/transformer_encoder_29/Encoder-1st-AdditionLayer-3/add:z:0;model_9/transformer_encoder_29/dropout_29/Identity:output:0*
T0*+
_output_shapes
:���������P	�
model_9/FinalLayerNorm/ShapeShapeBmodel_9/transformer_encoder_29/Encoder-2nd-AdditionLayer-3/add:z:0*
T0*
_output_shapes
::��t
*model_9/FinalLayerNorm/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: v
,model_9/FinalLayerNorm/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:v
,model_9/FinalLayerNorm/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
$model_9/FinalLayerNorm/strided_sliceStridedSlice%model_9/FinalLayerNorm/Shape:output:03model_9/FinalLayerNorm/strided_slice/stack:output:05model_9/FinalLayerNorm/strided_slice/stack_1:output:05model_9/FinalLayerNorm/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskf
model_9/FinalLayerNorm/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
model_9/FinalLayerNorm/ProdProd-model_9/FinalLayerNorm/strided_slice:output:0%model_9/FinalLayerNorm/Const:output:0*
T0*
_output_shapes
: v
,model_9/FinalLayerNorm/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:x
.model_9/FinalLayerNorm/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: x
.model_9/FinalLayerNorm/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
&model_9/FinalLayerNorm/strided_slice_1StridedSlice%model_9/FinalLayerNorm/Shape:output:05model_9/FinalLayerNorm/strided_slice_1/stack:output:07model_9/FinalLayerNorm/strided_slice_1/stack_1:output:07model_9/FinalLayerNorm/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_maskh
model_9/FinalLayerNorm/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
model_9/FinalLayerNorm/Prod_1Prod/model_9/FinalLayerNorm/strided_slice_1:output:0'model_9/FinalLayerNorm/Const_1:output:0*
T0*
_output_shapes
: h
&model_9/FinalLayerNorm/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :h
&model_9/FinalLayerNorm/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :�
$model_9/FinalLayerNorm/Reshape/shapePack/model_9/FinalLayerNorm/Reshape/shape/0:output:0$model_9/FinalLayerNorm/Prod:output:0&model_9/FinalLayerNorm/Prod_1:output:0/model_9/FinalLayerNorm/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:�
model_9/FinalLayerNorm/ReshapeReshapeBmodel_9/transformer_encoder_29/Encoder-2nd-AdditionLayer-3/add:z:0-model_9/FinalLayerNorm/Reshape/shape:output:0*
T0*8
_output_shapes&
$:"������������������~
"model_9/FinalLayerNorm/ones/packedPack$model_9/FinalLayerNorm/Prod:output:0*
N*
T0*
_output_shapes
:f
!model_9/FinalLayerNorm/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
model_9/FinalLayerNorm/onesFill+model_9/FinalLayerNorm/ones/packed:output:0*model_9/FinalLayerNorm/ones/Const:output:0*
T0*#
_output_shapes
:���������
#model_9/FinalLayerNorm/zeros/packedPack$model_9/FinalLayerNorm/Prod:output:0*
N*
T0*
_output_shapes
:g
"model_9/FinalLayerNorm/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
model_9/FinalLayerNorm/zerosFill,model_9/FinalLayerNorm/zeros/packed:output:0+model_9/FinalLayerNorm/zeros/Const:output:0*
T0*#
_output_shapes
:���������a
model_9/FinalLayerNorm/Const_2Const*
_output_shapes
: *
dtype0*
valueB a
model_9/FinalLayerNorm/Const_3Const*
_output_shapes
: *
dtype0*
valueB �
'model_9/FinalLayerNorm/FusedBatchNormV3FusedBatchNormV3'model_9/FinalLayerNorm/Reshape:output:0$model_9/FinalLayerNorm/ones:output:0%model_9/FinalLayerNorm/zeros:output:0'model_9/FinalLayerNorm/Const_2:output:0'model_9/FinalLayerNorm/Const_3:output:0*
T0*
U0*x
_output_shapesf
d:"������������������:���������:���������:���������:���������:*
data_formatNCHW*
epsilon%o�:�
 model_9/FinalLayerNorm/Reshape_1Reshape+model_9/FinalLayerNorm/FusedBatchNormV3:y:0%model_9/FinalLayerNorm/Shape:output:0*
T0*+
_output_shapes
:���������P	�
)model_9/FinalLayerNorm/mul/ReadVariableOpReadVariableOp2model_9_finallayernorm_mul_readvariableop_resource*
_output_shapes
:	*
dtype0�
model_9/FinalLayerNorm/mulMul)model_9/FinalLayerNorm/Reshape_1:output:01model_9/FinalLayerNorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������P	�
)model_9/FinalLayerNorm/add/ReadVariableOpReadVariableOp2model_9_finallayernorm_add_readvariableop_resource*
_output_shapes
:	*
dtype0�
model_9/FinalLayerNorm/addAddV2model_9/FinalLayerNorm/mul:z:01model_9/FinalLayerNorm/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������P	�
>model_9/ReduceStackDimensionViaSummation/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :�
,model_9/ReduceStackDimensionViaSummation/SumSummodel_9/FinalLayerNorm/add:z:0Gmodel_9/ReduceStackDimensionViaSummation/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:���������	w
2model_9/ReduceStackDimensionViaSummation/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �B�
0model_9/ReduceStackDimensionViaSummation/truedivRealDiv5model_9/ReduceStackDimensionViaSummation/Sum:output:0;model_9/ReduceStackDimensionViaSummation/truediv/y:output:0*
T0*'
_output_shapes
:���������	z
!model_9/StandardizeTimeLimit/CastCasttimelimitinput*

DstT0*

SrcT0*'
_output_shapes
:���������g
"model_9/StandardizeTimeLimit/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  pA�
 model_9/StandardizeTimeLimit/subSub%model_9/StandardizeTimeLimit/Cast:y:0+model_9/StandardizeTimeLimit/sub/y:output:0*
T0*'
_output_shapes
:���������k
&model_9/StandardizeTimeLimit/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �@�
$model_9/StandardizeTimeLimit/truedivRealDiv$model_9/StandardizeTimeLimit/sub:z:0/model_9/StandardizeTimeLimit/truediv/y:output:0*
T0*'
_output_shapes
:���������f
$model_9/ConcatenateLayer/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
model_9/ConcatenateLayer/concatConcatV24model_9/ReduceStackDimensionViaSummation/truediv:z:0(model_9/StandardizeTimeLimit/truediv:z:0-model_9/ConcatenateLayer/concat/axis:output:0*
N*
T0*'
_output_shapes
:���������
�
:model_9/FullyConnectedLayerAreaRatio/MatMul/ReadVariableOpReadVariableOpCmodel_9_fullyconnectedlayerarearatio_matmul_readvariableop_resource*
_output_shapes

:

*
dtype0�
+model_9/FullyConnectedLayerAreaRatio/MatMulMatMul(model_9/ConcatenateLayer/concat:output:0Bmodel_9/FullyConnectedLayerAreaRatio/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
�
;model_9/FullyConnectedLayerAreaRatio/BiasAdd/ReadVariableOpReadVariableOpDmodel_9_fullyconnectedlayerarearatio_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0�
,model_9/FullyConnectedLayerAreaRatio/BiasAddBiasAdd5model_9/FullyConnectedLayerAreaRatio/MatMul:product:0Cmodel_9/FullyConnectedLayerAreaRatio/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
t
/model_9/FullyConnectedLayerAreaRatio/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
-model_9/FullyConnectedLayerAreaRatio/Gelu/mulMul8model_9/FullyConnectedLayerAreaRatio/Gelu/mul/x:output:05model_9/FullyConnectedLayerAreaRatio/BiasAdd:output:0*
T0*'
_output_shapes
:���������
u
0model_9/FullyConnectedLayerAreaRatio/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *��?�
1model_9/FullyConnectedLayerAreaRatio/Gelu/truedivRealDiv5model_9/FullyConnectedLayerAreaRatio/BiasAdd:output:09model_9/FullyConnectedLayerAreaRatio/Gelu/Cast/x:output:0*
T0*'
_output_shapes
:���������
�
-model_9/FullyConnectedLayerAreaRatio/Gelu/ErfErf5model_9/FullyConnectedLayerAreaRatio/Gelu/truediv:z:0*
T0*'
_output_shapes
:���������
t
/model_9/FullyConnectedLayerAreaRatio/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
-model_9/FullyConnectedLayerAreaRatio/Gelu/addAddV28model_9/FullyConnectedLayerAreaRatio/Gelu/add/x:output:01model_9/FullyConnectedLayerAreaRatio/Gelu/Erf:y:0*
T0*'
_output_shapes
:���������
�
/model_9/FullyConnectedLayerAreaRatio/Gelu/mul_1Mul1model_9/FullyConnectedLayerAreaRatio/Gelu/mul:z:01model_9/FullyConnectedLayerAreaRatio/Gelu/add:z:0*
T0*'
_output_shapes
:���������
�
1model_9/PredictionAreaRatio/MatMul/ReadVariableOpReadVariableOp:model_9_predictionarearatio_matmul_readvariableop_resource*
_output_shapes

:
*
dtype0�
"model_9/PredictionAreaRatio/MatMulMatMul3model_9/FullyConnectedLayerAreaRatio/Gelu/mul_1:z:09model_9/PredictionAreaRatio/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
2model_9/PredictionAreaRatio/BiasAdd/ReadVariableOpReadVariableOp;model_9_predictionarearatio_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
#model_9/PredictionAreaRatio/BiasAddBiasAdd,model_9/PredictionAreaRatio/MatMul:product:0:model_9/PredictionAreaRatio/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
#model_9/PredictionAreaRatio/SigmoidSigmoid,model_9/PredictionAreaRatio/BiasAdd:output:0*
T0*'
_output_shapes
:����������
1model_9/PredictionStacks/Tensordot/ReadVariableOpReadVariableOp:model_9_predictionstacks_tensordot_readvariableop_resource*
_output_shapes

:	*
dtype0q
'model_9/PredictionStacks/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:x
'model_9/PredictionStacks/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       �
(model_9/PredictionStacks/Tensordot/ShapeShapemodel_9/FinalLayerNorm/add:z:0*
T0*
_output_shapes
::��r
0model_9/PredictionStacks/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
+model_9/PredictionStacks/Tensordot/GatherV2GatherV21model_9/PredictionStacks/Tensordot/Shape:output:00model_9/PredictionStacks/Tensordot/free:output:09model_9/PredictionStacks/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:t
2model_9/PredictionStacks/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
-model_9/PredictionStacks/Tensordot/GatherV2_1GatherV21model_9/PredictionStacks/Tensordot/Shape:output:00model_9/PredictionStacks/Tensordot/axes:output:0;model_9/PredictionStacks/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:r
(model_9/PredictionStacks/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
'model_9/PredictionStacks/Tensordot/ProdProd4model_9/PredictionStacks/Tensordot/GatherV2:output:01model_9/PredictionStacks/Tensordot/Const:output:0*
T0*
_output_shapes
: t
*model_9/PredictionStacks/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
)model_9/PredictionStacks/Tensordot/Prod_1Prod6model_9/PredictionStacks/Tensordot/GatherV2_1:output:03model_9/PredictionStacks/Tensordot/Const_1:output:0*
T0*
_output_shapes
: p
.model_9/PredictionStacks/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
)model_9/PredictionStacks/Tensordot/concatConcatV20model_9/PredictionStacks/Tensordot/free:output:00model_9/PredictionStacks/Tensordot/axes:output:07model_9/PredictionStacks/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
(model_9/PredictionStacks/Tensordot/stackPack0model_9/PredictionStacks/Tensordot/Prod:output:02model_9/PredictionStacks/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
,model_9/PredictionStacks/Tensordot/transpose	Transposemodel_9/FinalLayerNorm/add:z:02model_9/PredictionStacks/Tensordot/concat:output:0*
T0*+
_output_shapes
:���������P	�
*model_9/PredictionStacks/Tensordot/ReshapeReshape0model_9/PredictionStacks/Tensordot/transpose:y:01model_9/PredictionStacks/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
)model_9/PredictionStacks/Tensordot/MatMulMatMul3model_9/PredictionStacks/Tensordot/Reshape:output:09model_9/PredictionStacks/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������t
*model_9/PredictionStacks/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:r
0model_9/PredictionStacks/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
+model_9/PredictionStacks/Tensordot/concat_1ConcatV24model_9/PredictionStacks/Tensordot/GatherV2:output:03model_9/PredictionStacks/Tensordot/Const_2:output:09model_9/PredictionStacks/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
"model_9/PredictionStacks/TensordotReshape3model_9/PredictionStacks/Tensordot/MatMul:product:04model_9/PredictionStacks/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������P�
/model_9/PredictionStacks/BiasAdd/ReadVariableOpReadVariableOp8model_9_predictionstacks_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
 model_9/PredictionStacks/BiasAddBiasAdd+model_9/PredictionStacks/Tensordot:output:07model_9/PredictionStacks/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������P�
 model_9/PredictionStacks/SigmoidSigmoid)model_9/PredictionStacks/BiasAdd:output:0*
T0*+
_output_shapes
:���������Pm
model_9/Output/SqueezeSqueeze'model_9/PredictionAreaRatio/Sigmoid:y:0*
T0*
_output_shapes
:l
model_9/Output/Squeeze_1Squeeze$model_9/PredictionStacks/Sigmoid:y:0*
T0*
_output_shapes
:a
IdentityIdentity!model_9/Output/Squeeze_1:output:0^NoOp*
T0*
_output_shapes
:a

Identity_1Identitymodel_9/Output/Squeeze:output:0^NoOp*
T0*
_output_shapes
:�%
NoOpNoOp*^model_9/FinalLayerNorm/add/ReadVariableOp*^model_9/FinalLayerNorm/mul/ReadVariableOp<^model_9/FullyConnectedLayerAreaRatio/BiasAdd/ReadVariableOp;^model_9/FullyConnectedLayerAreaRatio/MatMul/ReadVariableOp3^model_9/PredictionAreaRatio/BiasAdd/ReadVariableOp2^model_9/PredictionAreaRatio/MatMul/ReadVariableOp0^model_9/PredictionStacks/BiasAdd/ReadVariableOp2^model_9/PredictionStacks/Tensordot/ReadVariableOpS^model_9/transformer_encoder_27/Encoder-1st-NormalizationLayer-1/add/ReadVariableOpS^model_9/transformer_encoder_27/Encoder-1st-NormalizationLayer-1/mul/ReadVariableOpS^model_9/transformer_encoder_27/Encoder-2nd-NormalizationLayer-1/add/ReadVariableOpS^model_9/transformer_encoder_27/Encoder-2nd-NormalizationLayer-1/mul/ReadVariableOpS^model_9/transformer_encoder_27/Encoder-FeedForwardLayer_1_1/BiasAdd/ReadVariableOpU^model_9/transformer_encoder_27/Encoder-FeedForwardLayer_1_1/Tensordot/ReadVariableOpS^model_9/transformer_encoder_27/Encoder-FeedForwardLayer_2_1/BiasAdd/ReadVariableOpU^model_9/transformer_encoder_27/Encoder-FeedForwardLayer_2_1/Tensordot/ReadVariableOp`^model_9/transformer_encoder_27/Encoder-SelfAttentionLayer-1/attention_output/add/ReadVariableOpj^model_9/transformer_encoder_27/Encoder-SelfAttentionLayer-1/attention_output/einsum/Einsum/ReadVariableOpS^model_9/transformer_encoder_27/Encoder-SelfAttentionLayer-1/key/add/ReadVariableOp]^model_9/transformer_encoder_27/Encoder-SelfAttentionLayer-1/key/einsum/Einsum/ReadVariableOpU^model_9/transformer_encoder_27/Encoder-SelfAttentionLayer-1/query/add/ReadVariableOp_^model_9/transformer_encoder_27/Encoder-SelfAttentionLayer-1/query/einsum/Einsum/ReadVariableOpU^model_9/transformer_encoder_27/Encoder-SelfAttentionLayer-1/value/add/ReadVariableOp_^model_9/transformer_encoder_27/Encoder-SelfAttentionLayer-1/value/einsum/Einsum/ReadVariableOpS^model_9/transformer_encoder_28/Encoder-1st-NormalizationLayer-2/add/ReadVariableOpS^model_9/transformer_encoder_28/Encoder-1st-NormalizationLayer-2/mul/ReadVariableOpS^model_9/transformer_encoder_28/Encoder-2nd-NormalizationLayer-2/add/ReadVariableOpS^model_9/transformer_encoder_28/Encoder-2nd-NormalizationLayer-2/mul/ReadVariableOpS^model_9/transformer_encoder_28/Encoder-FeedForwardLayer_1_2/BiasAdd/ReadVariableOpU^model_9/transformer_encoder_28/Encoder-FeedForwardLayer_1_2/Tensordot/ReadVariableOpS^model_9/transformer_encoder_28/Encoder-FeedForwardLayer_2_2/BiasAdd/ReadVariableOpU^model_9/transformer_encoder_28/Encoder-FeedForwardLayer_2_2/Tensordot/ReadVariableOp`^model_9/transformer_encoder_28/Encoder-SelfAttentionLayer-2/attention_output/add/ReadVariableOpj^model_9/transformer_encoder_28/Encoder-SelfAttentionLayer-2/attention_output/einsum/Einsum/ReadVariableOpS^model_9/transformer_encoder_28/Encoder-SelfAttentionLayer-2/key/add/ReadVariableOp]^model_9/transformer_encoder_28/Encoder-SelfAttentionLayer-2/key/einsum/Einsum/ReadVariableOpU^model_9/transformer_encoder_28/Encoder-SelfAttentionLayer-2/query/add/ReadVariableOp_^model_9/transformer_encoder_28/Encoder-SelfAttentionLayer-2/query/einsum/Einsum/ReadVariableOpU^model_9/transformer_encoder_28/Encoder-SelfAttentionLayer-2/value/add/ReadVariableOp_^model_9/transformer_encoder_28/Encoder-SelfAttentionLayer-2/value/einsum/Einsum/ReadVariableOpS^model_9/transformer_encoder_29/Encoder-1st-NormalizationLayer-3/add/ReadVariableOpS^model_9/transformer_encoder_29/Encoder-1st-NormalizationLayer-3/mul/ReadVariableOpS^model_9/transformer_encoder_29/Encoder-2nd-NormalizationLayer-3/add/ReadVariableOpS^model_9/transformer_encoder_29/Encoder-2nd-NormalizationLayer-3/mul/ReadVariableOpS^model_9/transformer_encoder_29/Encoder-FeedForwardLayer_1_3/BiasAdd/ReadVariableOpU^model_9/transformer_encoder_29/Encoder-FeedForwardLayer_1_3/Tensordot/ReadVariableOpS^model_9/transformer_encoder_29/Encoder-FeedForwardLayer_2_3/BiasAdd/ReadVariableOpU^model_9/transformer_encoder_29/Encoder-FeedForwardLayer_2_3/Tensordot/ReadVariableOp`^model_9/transformer_encoder_29/Encoder-SelfAttentionLayer-3/attention_output/add/ReadVariableOpj^model_9/transformer_encoder_29/Encoder-SelfAttentionLayer-3/attention_output/einsum/Einsum/ReadVariableOpS^model_9/transformer_encoder_29/Encoder-SelfAttentionLayer-3/key/add/ReadVariableOp]^model_9/transformer_encoder_29/Encoder-SelfAttentionLayer-3/key/einsum/Einsum/ReadVariableOpU^model_9/transformer_encoder_29/Encoder-SelfAttentionLayer-3/query/add/ReadVariableOp_^model_9/transformer_encoder_29/Encoder-SelfAttentionLayer-3/query/einsum/Einsum/ReadVariableOpU^model_9/transformer_encoder_29/Encoder-SelfAttentionLayer-3/value/add/ReadVariableOp_^model_9/transformer_encoder_29/Encoder-SelfAttentionLayer-3/value/einsum/Einsum/ReadVariableOp*
_output_shapes
 "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:���������P	:���������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2V
)model_9/FinalLayerNorm/add/ReadVariableOp)model_9/FinalLayerNorm/add/ReadVariableOp2V
)model_9/FinalLayerNorm/mul/ReadVariableOp)model_9/FinalLayerNorm/mul/ReadVariableOp2z
;model_9/FullyConnectedLayerAreaRatio/BiasAdd/ReadVariableOp;model_9/FullyConnectedLayerAreaRatio/BiasAdd/ReadVariableOp2x
:model_9/FullyConnectedLayerAreaRatio/MatMul/ReadVariableOp:model_9/FullyConnectedLayerAreaRatio/MatMul/ReadVariableOp2h
2model_9/PredictionAreaRatio/BiasAdd/ReadVariableOp2model_9/PredictionAreaRatio/BiasAdd/ReadVariableOp2f
1model_9/PredictionAreaRatio/MatMul/ReadVariableOp1model_9/PredictionAreaRatio/MatMul/ReadVariableOp2b
/model_9/PredictionStacks/BiasAdd/ReadVariableOp/model_9/PredictionStacks/BiasAdd/ReadVariableOp2f
1model_9/PredictionStacks/Tensordot/ReadVariableOp1model_9/PredictionStacks/Tensordot/ReadVariableOp2�
Rmodel_9/transformer_encoder_27/Encoder-1st-NormalizationLayer-1/add/ReadVariableOpRmodel_9/transformer_encoder_27/Encoder-1st-NormalizationLayer-1/add/ReadVariableOp2�
Rmodel_9/transformer_encoder_27/Encoder-1st-NormalizationLayer-1/mul/ReadVariableOpRmodel_9/transformer_encoder_27/Encoder-1st-NormalizationLayer-1/mul/ReadVariableOp2�
Rmodel_9/transformer_encoder_27/Encoder-2nd-NormalizationLayer-1/add/ReadVariableOpRmodel_9/transformer_encoder_27/Encoder-2nd-NormalizationLayer-1/add/ReadVariableOp2�
Rmodel_9/transformer_encoder_27/Encoder-2nd-NormalizationLayer-1/mul/ReadVariableOpRmodel_9/transformer_encoder_27/Encoder-2nd-NormalizationLayer-1/mul/ReadVariableOp2�
Rmodel_9/transformer_encoder_27/Encoder-FeedForwardLayer_1_1/BiasAdd/ReadVariableOpRmodel_9/transformer_encoder_27/Encoder-FeedForwardLayer_1_1/BiasAdd/ReadVariableOp2�
Tmodel_9/transformer_encoder_27/Encoder-FeedForwardLayer_1_1/Tensordot/ReadVariableOpTmodel_9/transformer_encoder_27/Encoder-FeedForwardLayer_1_1/Tensordot/ReadVariableOp2�
Rmodel_9/transformer_encoder_27/Encoder-FeedForwardLayer_2_1/BiasAdd/ReadVariableOpRmodel_9/transformer_encoder_27/Encoder-FeedForwardLayer_2_1/BiasAdd/ReadVariableOp2�
Tmodel_9/transformer_encoder_27/Encoder-FeedForwardLayer_2_1/Tensordot/ReadVariableOpTmodel_9/transformer_encoder_27/Encoder-FeedForwardLayer_2_1/Tensordot/ReadVariableOp2�
_model_9/transformer_encoder_27/Encoder-SelfAttentionLayer-1/attention_output/add/ReadVariableOp_model_9/transformer_encoder_27/Encoder-SelfAttentionLayer-1/attention_output/add/ReadVariableOp2�
imodel_9/transformer_encoder_27/Encoder-SelfAttentionLayer-1/attention_output/einsum/Einsum/ReadVariableOpimodel_9/transformer_encoder_27/Encoder-SelfAttentionLayer-1/attention_output/einsum/Einsum/ReadVariableOp2�
Rmodel_9/transformer_encoder_27/Encoder-SelfAttentionLayer-1/key/add/ReadVariableOpRmodel_9/transformer_encoder_27/Encoder-SelfAttentionLayer-1/key/add/ReadVariableOp2�
\model_9/transformer_encoder_27/Encoder-SelfAttentionLayer-1/key/einsum/Einsum/ReadVariableOp\model_9/transformer_encoder_27/Encoder-SelfAttentionLayer-1/key/einsum/Einsum/ReadVariableOp2�
Tmodel_9/transformer_encoder_27/Encoder-SelfAttentionLayer-1/query/add/ReadVariableOpTmodel_9/transformer_encoder_27/Encoder-SelfAttentionLayer-1/query/add/ReadVariableOp2�
^model_9/transformer_encoder_27/Encoder-SelfAttentionLayer-1/query/einsum/Einsum/ReadVariableOp^model_9/transformer_encoder_27/Encoder-SelfAttentionLayer-1/query/einsum/Einsum/ReadVariableOp2�
Tmodel_9/transformer_encoder_27/Encoder-SelfAttentionLayer-1/value/add/ReadVariableOpTmodel_9/transformer_encoder_27/Encoder-SelfAttentionLayer-1/value/add/ReadVariableOp2�
^model_9/transformer_encoder_27/Encoder-SelfAttentionLayer-1/value/einsum/Einsum/ReadVariableOp^model_9/transformer_encoder_27/Encoder-SelfAttentionLayer-1/value/einsum/Einsum/ReadVariableOp2�
Rmodel_9/transformer_encoder_28/Encoder-1st-NormalizationLayer-2/add/ReadVariableOpRmodel_9/transformer_encoder_28/Encoder-1st-NormalizationLayer-2/add/ReadVariableOp2�
Rmodel_9/transformer_encoder_28/Encoder-1st-NormalizationLayer-2/mul/ReadVariableOpRmodel_9/transformer_encoder_28/Encoder-1st-NormalizationLayer-2/mul/ReadVariableOp2�
Rmodel_9/transformer_encoder_28/Encoder-2nd-NormalizationLayer-2/add/ReadVariableOpRmodel_9/transformer_encoder_28/Encoder-2nd-NormalizationLayer-2/add/ReadVariableOp2�
Rmodel_9/transformer_encoder_28/Encoder-2nd-NormalizationLayer-2/mul/ReadVariableOpRmodel_9/transformer_encoder_28/Encoder-2nd-NormalizationLayer-2/mul/ReadVariableOp2�
Rmodel_9/transformer_encoder_28/Encoder-FeedForwardLayer_1_2/BiasAdd/ReadVariableOpRmodel_9/transformer_encoder_28/Encoder-FeedForwardLayer_1_2/BiasAdd/ReadVariableOp2�
Tmodel_9/transformer_encoder_28/Encoder-FeedForwardLayer_1_2/Tensordot/ReadVariableOpTmodel_9/transformer_encoder_28/Encoder-FeedForwardLayer_1_2/Tensordot/ReadVariableOp2�
Rmodel_9/transformer_encoder_28/Encoder-FeedForwardLayer_2_2/BiasAdd/ReadVariableOpRmodel_9/transformer_encoder_28/Encoder-FeedForwardLayer_2_2/BiasAdd/ReadVariableOp2�
Tmodel_9/transformer_encoder_28/Encoder-FeedForwardLayer_2_2/Tensordot/ReadVariableOpTmodel_9/transformer_encoder_28/Encoder-FeedForwardLayer_2_2/Tensordot/ReadVariableOp2�
_model_9/transformer_encoder_28/Encoder-SelfAttentionLayer-2/attention_output/add/ReadVariableOp_model_9/transformer_encoder_28/Encoder-SelfAttentionLayer-2/attention_output/add/ReadVariableOp2�
imodel_9/transformer_encoder_28/Encoder-SelfAttentionLayer-2/attention_output/einsum/Einsum/ReadVariableOpimodel_9/transformer_encoder_28/Encoder-SelfAttentionLayer-2/attention_output/einsum/Einsum/ReadVariableOp2�
Rmodel_9/transformer_encoder_28/Encoder-SelfAttentionLayer-2/key/add/ReadVariableOpRmodel_9/transformer_encoder_28/Encoder-SelfAttentionLayer-2/key/add/ReadVariableOp2�
\model_9/transformer_encoder_28/Encoder-SelfAttentionLayer-2/key/einsum/Einsum/ReadVariableOp\model_9/transformer_encoder_28/Encoder-SelfAttentionLayer-2/key/einsum/Einsum/ReadVariableOp2�
Tmodel_9/transformer_encoder_28/Encoder-SelfAttentionLayer-2/query/add/ReadVariableOpTmodel_9/transformer_encoder_28/Encoder-SelfAttentionLayer-2/query/add/ReadVariableOp2�
^model_9/transformer_encoder_28/Encoder-SelfAttentionLayer-2/query/einsum/Einsum/ReadVariableOp^model_9/transformer_encoder_28/Encoder-SelfAttentionLayer-2/query/einsum/Einsum/ReadVariableOp2�
Tmodel_9/transformer_encoder_28/Encoder-SelfAttentionLayer-2/value/add/ReadVariableOpTmodel_9/transformer_encoder_28/Encoder-SelfAttentionLayer-2/value/add/ReadVariableOp2�
^model_9/transformer_encoder_28/Encoder-SelfAttentionLayer-2/value/einsum/Einsum/ReadVariableOp^model_9/transformer_encoder_28/Encoder-SelfAttentionLayer-2/value/einsum/Einsum/ReadVariableOp2�
Rmodel_9/transformer_encoder_29/Encoder-1st-NormalizationLayer-3/add/ReadVariableOpRmodel_9/transformer_encoder_29/Encoder-1st-NormalizationLayer-3/add/ReadVariableOp2�
Rmodel_9/transformer_encoder_29/Encoder-1st-NormalizationLayer-3/mul/ReadVariableOpRmodel_9/transformer_encoder_29/Encoder-1st-NormalizationLayer-3/mul/ReadVariableOp2�
Rmodel_9/transformer_encoder_29/Encoder-2nd-NormalizationLayer-3/add/ReadVariableOpRmodel_9/transformer_encoder_29/Encoder-2nd-NormalizationLayer-3/add/ReadVariableOp2�
Rmodel_9/transformer_encoder_29/Encoder-2nd-NormalizationLayer-3/mul/ReadVariableOpRmodel_9/transformer_encoder_29/Encoder-2nd-NormalizationLayer-3/mul/ReadVariableOp2�
Rmodel_9/transformer_encoder_29/Encoder-FeedForwardLayer_1_3/BiasAdd/ReadVariableOpRmodel_9/transformer_encoder_29/Encoder-FeedForwardLayer_1_3/BiasAdd/ReadVariableOp2�
Tmodel_9/transformer_encoder_29/Encoder-FeedForwardLayer_1_3/Tensordot/ReadVariableOpTmodel_9/transformer_encoder_29/Encoder-FeedForwardLayer_1_3/Tensordot/ReadVariableOp2�
Rmodel_9/transformer_encoder_29/Encoder-FeedForwardLayer_2_3/BiasAdd/ReadVariableOpRmodel_9/transformer_encoder_29/Encoder-FeedForwardLayer_2_3/BiasAdd/ReadVariableOp2�
Tmodel_9/transformer_encoder_29/Encoder-FeedForwardLayer_2_3/Tensordot/ReadVariableOpTmodel_9/transformer_encoder_29/Encoder-FeedForwardLayer_2_3/Tensordot/ReadVariableOp2�
_model_9/transformer_encoder_29/Encoder-SelfAttentionLayer-3/attention_output/add/ReadVariableOp_model_9/transformer_encoder_29/Encoder-SelfAttentionLayer-3/attention_output/add/ReadVariableOp2�
imodel_9/transformer_encoder_29/Encoder-SelfAttentionLayer-3/attention_output/einsum/Einsum/ReadVariableOpimodel_9/transformer_encoder_29/Encoder-SelfAttentionLayer-3/attention_output/einsum/Einsum/ReadVariableOp2�
Rmodel_9/transformer_encoder_29/Encoder-SelfAttentionLayer-3/key/add/ReadVariableOpRmodel_9/transformer_encoder_29/Encoder-SelfAttentionLayer-3/key/add/ReadVariableOp2�
\model_9/transformer_encoder_29/Encoder-SelfAttentionLayer-3/key/einsum/Einsum/ReadVariableOp\model_9/transformer_encoder_29/Encoder-SelfAttentionLayer-3/key/einsum/Einsum/ReadVariableOp2�
Tmodel_9/transformer_encoder_29/Encoder-SelfAttentionLayer-3/query/add/ReadVariableOpTmodel_9/transformer_encoder_29/Encoder-SelfAttentionLayer-3/query/add/ReadVariableOp2�
^model_9/transformer_encoder_29/Encoder-SelfAttentionLayer-3/query/einsum/Einsum/ReadVariableOp^model_9/transformer_encoder_29/Encoder-SelfAttentionLayer-3/query/einsum/Einsum/ReadVariableOp2�
Tmodel_9/transformer_encoder_29/Encoder-SelfAttentionLayer-3/value/add/ReadVariableOpTmodel_9/transformer_encoder_29/Encoder-SelfAttentionLayer-3/value/add/ReadVariableOp2�
^model_9/transformer_encoder_29/Encoder-SelfAttentionLayer-3/value/einsum/Einsum/ReadVariableOp^model_9/transformer_encoder_29/Encoder-SelfAttentionLayer-3/value/einsum/Einsum/ReadVariableOp:(9$
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
��
�
S__inference_transformer_encoder_28_layer_call_and_return_conditional_losses_2631110

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
dropout_28/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?�
dropout_28/dropout/MulMul-Encoder-FeedForwardLayer_2_2/BiasAdd:output:0!dropout_28/dropout/Const:output:0*
T0*+
_output_shapes
:���������P	�
dropout_28/dropout/ShapeShape-Encoder-FeedForwardLayer_2_2/BiasAdd:output:0*
T0*
_output_shapes
::���
/dropout_28/dropout/random_uniform/RandomUniformRandomUniform!dropout_28/dropout/Shape:output:0*
T0*+
_output_shapes
:���������P	*
dtype0*
seed2*
seed��f
!dropout_28/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=�
dropout_28/dropout/GreaterEqualGreaterEqual8dropout_28/dropout/random_uniform/RandomUniform:output:0*dropout_28/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:���������P	_
dropout_28/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout_28/dropout/SelectV2SelectV2#dropout_28/dropout/GreaterEqual:z:0dropout_28/dropout/Mul:z:0#dropout_28/dropout/Const_1:output:0*
T0*+
_output_shapes
:���������P	�
Encoder-2nd-AdditionLayer-2/addAddV2#Encoder-1st-AdditionLayer-2/add:z:0$dropout_28/dropout/SelectV2:output:0*
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
 
_user_specified_nameinputs"�L
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
StatefulPartitionedCall:0tensorflow/serving/predict:́
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
layer-13
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures"
_tf_keras_network
"
_tf_keras_input_layer
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
�
	variables
trainable_variables
regularization_losses
 	keras_api
!__call__
*"&call_and_return_all_conditional_losses
#self_attention_layer
$add1
%add2
&
layernorm1
'
layernorm2
(feed_forward_layer_1
)feed_forward_layer_2
*dropout_layer"
_tf_keras_layer
�
+	variables
,trainable_variables
-regularization_losses
.	keras_api
/__call__
*0&call_and_return_all_conditional_losses
1self_attention_layer
2add1
3add2
4
layernorm1
5
layernorm2
6feed_forward_layer_1
7feed_forward_layer_2
8dropout_layer"
_tf_keras_layer
�
9	variables
:trainable_variables
;regularization_losses
<	keras_api
=__call__
*>&call_and_return_all_conditional_losses
?self_attention_layer
@add1
Aadd2
B
layernorm1
C
layernorm2
Dfeed_forward_layer_1
Efeed_forward_layer_2
Fdropout_layer"
_tf_keras_layer
�
G	variables
Htrainable_variables
Iregularization_losses
J	keras_api
K__call__
*L&call_and_return_all_conditional_losses
Maxis
	Ngamma
Obeta"
_tf_keras_layer
"
_tf_keras_input_layer
�
P	variables
Qtrainable_variables
Rregularization_losses
S	keras_api
T__call__
*U&call_and_return_all_conditional_losses"
_tf_keras_layer
�
V	variables
Wtrainable_variables
Xregularization_losses
Y	keras_api
Z__call__
*[&call_and_return_all_conditional_losses"
_tf_keras_layer
�
\	variables
]trainable_variables
^regularization_losses
_	keras_api
`__call__
*a&call_and_return_all_conditional_losses"
_tf_keras_layer
�
b	variables
ctrainable_variables
dregularization_losses
e	keras_api
f__call__
*g&call_and_return_all_conditional_losses

hkernel
ibias"
_tf_keras_layer
�
j	variables
ktrainable_variables
lregularization_losses
m	keras_api
n__call__
*o&call_and_return_all_conditional_losses

pkernel
qbias"
_tf_keras_layer
�
r	variables
strainable_variables
tregularization_losses
u	keras_api
v__call__
*w&call_and_return_all_conditional_losses

xkernel
ybias"
_tf_keras_layer
�
z	variables
{trainable_variables
|regularization_losses
}	keras_api
~__call__
*&call_and_return_all_conditional_losses"
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
N48
O49
h50
i51
p52
q53
x54
y55"
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
N48
O49
h50
i51
p52
q53
x54
y55"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
)__inference_model_9_layer_call_fn_2629826
)__inference_model_9_layer_call_fn_2629946�
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
D__inference_model_9_layer_call_and_return_conditional_losses_2629022
D__inference_model_9_layer_call_and_return_conditional_losses_2629706�
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
"__inference__wrapped_model_2628176StackLevelInputFeaturesTimeLimitInput"�
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
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
.__inference_MaskingLayer_layer_call_fn_2630393�
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
I__inference_MaskingLayer_layer_call_and_return_conditional_losses_2630404�
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
	variables
trainable_variables
regularization_losses
!__call__
*"&call_and_return_all_conditional_losses
&""call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
8__inference_transformer_encoder_27_layer_call_fn_2630443
8__inference_transformer_encoder_27_layer_call_fn_2630482�
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
S__inference_transformer_encoder_27_layer_call_and_return_conditional_losses_2630670
S__inference_transformer_encoder_27_layer_call_and_return_conditional_losses_2630844�
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
+	variables
,trainable_variables
-regularization_losses
/__call__
*0&call_and_return_all_conditional_losses
&0"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
8__inference_transformer_encoder_28_layer_call_fn_2630883
8__inference_transformer_encoder_28_layer_call_fn_2630922�
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
S__inference_transformer_encoder_28_layer_call_and_return_conditional_losses_2631110
S__inference_transformer_encoder_28_layer_call_and_return_conditional_losses_2631284�
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
9	variables
:trainable_variables
;regularization_losses
=__call__
*>&call_and_return_all_conditional_losses
&>"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
8__inference_transformer_encoder_29_layer_call_fn_2631323
8__inference_transformer_encoder_29_layer_call_fn_2631362�
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
S__inference_transformer_encoder_29_layer_call_and_return_conditional_losses_2631550
S__inference_transformer_encoder_29_layer_call_and_return_conditional_losses_2631724�
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
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�_random_generator"
_tf_keras_layer
.
N0
O1"
trackable_list_wrapper
.
N0
O1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
G	variables
Htrainable_variables
Iregularization_losses
K__call__
*L&call_and_return_all_conditional_losses
&L"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
0__inference_FinalLayerNorm_layer_call_fn_2631733�
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
K__inference_FinalLayerNorm_layer_call_and_return_conditional_losses_2631775�
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
P	variables
Qtrainable_variables
Rregularization_losses
T__call__
*U&call_and_return_all_conditional_losses
&U"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
B__inference_ReduceStackDimensionViaSummation_layer_call_fn_2631780
B__inference_ReduceStackDimensionViaSummation_layer_call_fn_2631785�
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
]__inference_ReduceStackDimensionViaSummation_layer_call_and_return_conditional_losses_2631793
]__inference_ReduceStackDimensionViaSummation_layer_call_and_return_conditional_losses_2631801�
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
V	variables
Wtrainable_variables
Xregularization_losses
Z__call__
*[&call_and_return_all_conditional_losses
&["call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
6__inference_StandardizeTimeLimit_layer_call_fn_2631806
6__inference_StandardizeTimeLimit_layer_call_fn_2631811�
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
Q__inference_StandardizeTimeLimit_layer_call_and_return_conditional_losses_2631819
Q__inference_StandardizeTimeLimit_layer_call_and_return_conditional_losses_2631827�
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
\	variables
]trainable_variables
^regularization_losses
`__call__
*a&call_and_return_all_conditional_losses
&a"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
2__inference_ConcatenateLayer_layer_call_fn_2631833�
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
M__inference_ConcatenateLayer_layer_call_and_return_conditional_losses_2631840�
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
h0
i1"
trackable_list_wrapper
.
h0
i1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
b	variables
ctrainable_variables
dregularization_losses
f__call__
*g&call_and_return_all_conditional_losses
&g"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
>__inference_FullyConnectedLayerAreaRatio_layer_call_fn_2631849�
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
Y__inference_FullyConnectedLayerAreaRatio_layer_call_and_return_conditional_losses_2631867�
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
p0
q1"
trackable_list_wrapper
.
p0
q1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
j	variables
ktrainable_variables
lregularization_losses
n__call__
*o&call_and_return_all_conditional_losses
&o"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
2__inference_PredictionStacks_layer_call_fn_2631876�
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
M__inference_PredictionStacks_layer_call_and_return_conditional_losses_2631907�
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
):'	2PredictionStacks/kernel
#:!2PredictionStacks/bias
.
x0
y1"
trackable_list_wrapper
.
x0
y1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
r	variables
strainable_variables
tregularization_losses
v__call__
*w&call_and_return_all_conditional_losses
&w"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
5__inference_PredictionAreaRatio_layer_call_fn_2631916�
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
P__inference_PredictionAreaRatio_layer_call_and_return_conditional_losses_2631927�
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
z	variables
{trainable_variables
|regularization_losses
~__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_1
�trace_2
�trace_32�
(__inference_Output_layer_call_fn_2631932
(__inference_Output_layer_call_fn_2631937
(__inference_Output_layer_call_fn_2631942
(__inference_Output_layer_call_fn_2631947�
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
 z�trace_0z�trace_1z�trace_2z�trace_3
�
�trace_0
�trace_1
�trace_2
�trace_32�
C__inference_Output_layer_call_and_return_conditional_losses_2631952
C__inference_Output_layer_call_and_return_conditional_losses_2631957
C__inference_Output_layer_call_and_return_conditional_losses_2631962
C__inference_Output_layer_call_and_return_conditional_losses_2631967�
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
 z�trace_0z�trace_1z�trace_2z�trace_3
V:T	2@transformer_encoder_27/Encoder-SelfAttentionLayer-1/query/kernel
P:N2>transformer_encoder_27/Encoder-SelfAttentionLayer-1/query/bias
T:R	2>transformer_encoder_27/Encoder-SelfAttentionLayer-1/key/kernel
N:L2<transformer_encoder_27/Encoder-SelfAttentionLayer-1/key/bias
V:T	2@transformer_encoder_27/Encoder-SelfAttentionLayer-1/value/kernel
P:N2>transformer_encoder_27/Encoder-SelfAttentionLayer-1/value/bias
a:_	2Ktransformer_encoder_27/Encoder-SelfAttentionLayer-1/attention_output/kernel
W:U	2Itransformer_encoder_27/Encoder-SelfAttentionLayer-1/attention_output/bias
K:I	2=transformer_encoder_27/Encoder-1st-NormalizationLayer-1/gamma
J:H	2<transformer_encoder_27/Encoder-1st-NormalizationLayer-1/beta
K:I	2=transformer_encoder_27/Encoder-2nd-NormalizationLayer-1/gamma
J:H	2<transformer_encoder_27/Encoder-2nd-NormalizationLayer-1/beta
L:J		2:transformer_encoder_27/Encoder-FeedForwardLayer_1_1/kernel
F:D	28transformer_encoder_27/Encoder-FeedForwardLayer_1_1/bias
L:J		2:transformer_encoder_27/Encoder-FeedForwardLayer_2_1/kernel
F:D	28transformer_encoder_27/Encoder-FeedForwardLayer_2_1/bias
V:T	2@transformer_encoder_28/Encoder-SelfAttentionLayer-2/query/kernel
P:N2>transformer_encoder_28/Encoder-SelfAttentionLayer-2/query/bias
T:R	2>transformer_encoder_28/Encoder-SelfAttentionLayer-2/key/kernel
N:L2<transformer_encoder_28/Encoder-SelfAttentionLayer-2/key/bias
V:T	2@transformer_encoder_28/Encoder-SelfAttentionLayer-2/value/kernel
P:N2>transformer_encoder_28/Encoder-SelfAttentionLayer-2/value/bias
a:_	2Ktransformer_encoder_28/Encoder-SelfAttentionLayer-2/attention_output/kernel
W:U	2Itransformer_encoder_28/Encoder-SelfAttentionLayer-2/attention_output/bias
K:I	2=transformer_encoder_28/Encoder-1st-NormalizationLayer-2/gamma
J:H	2<transformer_encoder_28/Encoder-1st-NormalizationLayer-2/beta
K:I	2=transformer_encoder_28/Encoder-2nd-NormalizationLayer-2/gamma
J:H	2<transformer_encoder_28/Encoder-2nd-NormalizationLayer-2/beta
L:J		2:transformer_encoder_28/Encoder-FeedForwardLayer_1_2/kernel
F:D	28transformer_encoder_28/Encoder-FeedForwardLayer_1_2/bias
L:J		2:transformer_encoder_28/Encoder-FeedForwardLayer_2_2/kernel
F:D	28transformer_encoder_28/Encoder-FeedForwardLayer_2_2/bias
V:T	2@transformer_encoder_29/Encoder-SelfAttentionLayer-3/query/kernel
P:N2>transformer_encoder_29/Encoder-SelfAttentionLayer-3/query/bias
T:R	2>transformer_encoder_29/Encoder-SelfAttentionLayer-3/key/kernel
N:L2<transformer_encoder_29/Encoder-SelfAttentionLayer-3/key/bias
V:T	2@transformer_encoder_29/Encoder-SelfAttentionLayer-3/value/kernel
P:N2>transformer_encoder_29/Encoder-SelfAttentionLayer-3/value/bias
a:_	2Ktransformer_encoder_29/Encoder-SelfAttentionLayer-3/attention_output/kernel
W:U	2Itransformer_encoder_29/Encoder-SelfAttentionLayer-3/attention_output/bias
K:I	2=transformer_encoder_29/Encoder-1st-NormalizationLayer-3/gamma
J:H	2<transformer_encoder_29/Encoder-1st-NormalizationLayer-3/beta
K:I	2=transformer_encoder_29/Encoder-2nd-NormalizationLayer-3/gamma
J:H	2<transformer_encoder_29/Encoder-2nd-NormalizationLayer-3/beta
L:J		2:transformer_encoder_29/Encoder-FeedForwardLayer_1_3/kernel
F:D	28transformer_encoder_29/Encoder-FeedForwardLayer_1_3/bias
L:J		2:transformer_encoder_29/Encoder-FeedForwardLayer_2_3/kernel
F:D	28transformer_encoder_29/Encoder-FeedForwardLayer_2_3/bias
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
13"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
)__inference_model_9_layer_call_fn_2629826StackLevelInputFeaturesTimeLimitInput"�
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
)__inference_model_9_layer_call_fn_2629946StackLevelInputFeaturesTimeLimitInput"�
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
D__inference_model_9_layer_call_and_return_conditional_losses_2629022StackLevelInputFeaturesTimeLimitInput"�
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
D__inference_model_9_layer_call_and_return_conditional_losses_2629706StackLevelInputFeaturesTimeLimitInput"�
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
%__inference_signature_wrapper_2630388StackLevelInputFeaturesTimeLimitInput"�
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
.__inference_MaskingLayer_layer_call_fn_2630393inputs"�
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
I__inference_MaskingLayer_layer_call_and_return_conditional_losses_2630404inputs"�
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
#0
$1
%2
&3
'4
(5
)6
*7"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
8__inference_transformer_encoder_27_layer_call_fn_2630443inputs"�
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
8__inference_transformer_encoder_27_layer_call_fn_2630482inputs"�
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
S__inference_transformer_encoder_27_layer_call_and_return_conditional_losses_2630670inputs"�
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
S__inference_transformer_encoder_27_layer_call_and_return_conditional_losses_2630844inputs"�
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
10
21
32
43
54
65
76
87"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
8__inference_transformer_encoder_28_layer_call_fn_2630883inputs"�
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
8__inference_transformer_encoder_28_layer_call_fn_2630922inputs"�
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
S__inference_transformer_encoder_28_layer_call_and_return_conditional_losses_2631110inputs"�
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
S__inference_transformer_encoder_28_layer_call_and_return_conditional_losses_2631284inputs"�
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
?0
@1
A2
B3
C4
D5
E6
F7"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
8__inference_transformer_encoder_29_layer_call_fn_2631323inputs"�
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
8__inference_transformer_encoder_29_layer_call_fn_2631362inputs"�
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
S__inference_transformer_encoder_29_layer_call_and_return_conditional_losses_2631550inputs"�
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
S__inference_transformer_encoder_29_layer_call_and_return_conditional_losses_2631724inputs"�
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
0__inference_FinalLayerNorm_layer_call_fn_2631733inputs"�
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
K__inference_FinalLayerNorm_layer_call_and_return_conditional_losses_2631775inputs"�
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
B__inference_ReduceStackDimensionViaSummation_layer_call_fn_2631780inputs"�
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
B__inference_ReduceStackDimensionViaSummation_layer_call_fn_2631785inputs"�
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
]__inference_ReduceStackDimensionViaSummation_layer_call_and_return_conditional_losses_2631793inputs"�
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
]__inference_ReduceStackDimensionViaSummation_layer_call_and_return_conditional_losses_2631801inputs"�
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
6__inference_StandardizeTimeLimit_layer_call_fn_2631806inputs"�
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
6__inference_StandardizeTimeLimit_layer_call_fn_2631811inputs"�
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
Q__inference_StandardizeTimeLimit_layer_call_and_return_conditional_losses_2631819inputs"�
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
Q__inference_StandardizeTimeLimit_layer_call_and_return_conditional_losses_2631827inputs"�
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
2__inference_ConcatenateLayer_layer_call_fn_2631833inputs_0inputs_1"�
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
M__inference_ConcatenateLayer_layer_call_and_return_conditional_losses_2631840inputs_0inputs_1"�
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
>__inference_FullyConnectedLayerAreaRatio_layer_call_fn_2631849inputs"�
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
Y__inference_FullyConnectedLayerAreaRatio_layer_call_and_return_conditional_losses_2631867inputs"�
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
2__inference_PredictionStacks_layer_call_fn_2631876inputs"�
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
M__inference_PredictionStacks_layer_call_and_return_conditional_losses_2631907inputs"�
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
5__inference_PredictionAreaRatio_layer_call_fn_2631916inputs"�
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
P__inference_PredictionAreaRatio_layer_call_and_return_conditional_losses_2631927inputs"�
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
(__inference_Output_layer_call_fn_2631932inputs"�
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
(__inference_Output_layer_call_fn_2631937inputs"�
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
(__inference_Output_layer_call_fn_2631942inputs"�
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
(__inference_Output_layer_call_fn_2631947inputs"�
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
C__inference_Output_layer_call_and_return_conditional_losses_2631952inputs"�
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
C__inference_Output_layer_call_and_return_conditional_losses_2631957inputs"�
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
C__inference_Output_layer_call_and_return_conditional_losses_2631962inputs"�
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
C__inference_Output_layer_call_and_return_conditional_losses_2631967inputs"�
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
M__inference_ConcatenateLayer_layer_call_and_return_conditional_losses_2631840�Z�W
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
2__inference_ConcatenateLayer_layer_call_fn_2631833Z�W
P�M
K�H
"�
inputs_0���������	
"�
inputs_1���������
� "!�
unknown���������
�
K__inference_FinalLayerNorm_layer_call_and_return_conditional_losses_2631775kNO3�0
)�&
$�!
inputs���������P	
� "0�-
&�#
tensor_0���������P	
� �
0__inference_FinalLayerNorm_layer_call_fn_2631733`NO3�0
)�&
$�!
inputs���������P	
� "%�"
unknown���������P	�
Y__inference_FullyConnectedLayerAreaRatio_layer_call_and_return_conditional_losses_2631867chi/�,
%�"
 �
inputs���������

� ",�)
"�
tensor_0���������

� �
>__inference_FullyConnectedLayerAreaRatio_layer_call_fn_2631849Xhi/�,
%�"
 �
inputs���������

� "!�
unknown���������
�
I__inference_MaskingLayer_layer_call_and_return_conditional_losses_2630404g3�0
)�&
$�!
inputs���������P	
� "0�-
&�#
tensor_0���������P	
� �
.__inference_MaskingLayer_layer_call_fn_2630393\3�0
)�&
$�!
inputs���������P	
� "%�"
unknown���������P	�
C__inference_Output_layer_call_and_return_conditional_losses_2631952X7�4
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
C__inference_Output_layer_call_and_return_conditional_losses_2631957\;�8
1�.
$�!
inputs���������P

 
p
� "�
�
tensor_0
� �
C__inference_Output_layer_call_and_return_conditional_losses_2631962X7�4
-�*
 �
inputs���������

 
p 
� "�
�
tensor_0
� �
C__inference_Output_layer_call_and_return_conditional_losses_2631967\;�8
1�.
$�!
inputs���������P

 
p 
� "�
�
tensor_0
� y
(__inference_Output_layer_call_fn_2631932M7�4
-�*
 �
inputs���������

 
p
� "�
unknowny
(__inference_Output_layer_call_fn_2631937M7�4
-�*
 �
inputs���������

 
p 
� "�
unknown}
(__inference_Output_layer_call_fn_2631942Q;�8
1�.
$�!
inputs���������P

 
p
� "�
unknown}
(__inference_Output_layer_call_fn_2631947Q;�8
1�.
$�!
inputs���������P

 
p 
� "�
unknown�
P__inference_PredictionAreaRatio_layer_call_and_return_conditional_losses_2631927cxy/�,
%�"
 �
inputs���������

� ",�)
"�
tensor_0���������
� �
5__inference_PredictionAreaRatio_layer_call_fn_2631916Xxy/�,
%�"
 �
inputs���������

� "!�
unknown����������
M__inference_PredictionStacks_layer_call_and_return_conditional_losses_2631907kpq3�0
)�&
$�!
inputs���������P	
� "0�-
&�#
tensor_0���������P
� �
2__inference_PredictionStacks_layer_call_fn_2631876`pq3�0
)�&
$�!
inputs���������P	
� "%�"
unknown���������P�
]__inference_ReduceStackDimensionViaSummation_layer_call_and_return_conditional_losses_2631793k;�8
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
]__inference_ReduceStackDimensionViaSummation_layer_call_and_return_conditional_losses_2631801k;�8
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
B__inference_ReduceStackDimensionViaSummation_layer_call_fn_2631780`;�8
1�.
$�!
inputs���������P	

 
p
� "!�
unknown���������	�
B__inference_ReduceStackDimensionViaSummation_layer_call_fn_2631785`;�8
1�.
$�!
inputs���������P	

 
p 
� "!�
unknown���������	�
Q__inference_StandardizeTimeLimit_layer_call_and_return_conditional_losses_2631819g7�4
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
Q__inference_StandardizeTimeLimit_layer_call_and_return_conditional_losses_2631827g7�4
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
6__inference_StandardizeTimeLimit_layer_call_fn_2631806\7�4
-�*
 �
inputs���������

 
p
� "!�
unknown����������
6__inference_StandardizeTimeLimit_layer_call_fn_2631811\7�4
-�*
 �
inputs���������

 
p 
� "!�
unknown����������
"__inference__wrapped_model_2628176�h������������������������������������������������NOhixypqs�p
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
D__inference_model_9_layer_call_and_return_conditional_losses_2629022�h������������������������������������������������NOhixypq{�x
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
D__inference_model_9_layer_call_and_return_conditional_losses_2629706�h������������������������������������������������NOhixypq{�x
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
)__inference_model_9_layer_call_fn_2629826�h������������������������������������������������NOhixypq{�x
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
)__inference_model_9_layer_call_fn_2629946�h������������������������������������������������NOhixypq{�x
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
%__inference_signature_wrapper_2630388�h������������������������������������������������NOhixypq���
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
S__inference_transformer_encoder_27_layer_call_and_return_conditional_losses_2630670� ����������������C�@
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
S__inference_transformer_encoder_27_layer_call_and_return_conditional_losses_2630844� ����������������C�@
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
8__inference_transformer_encoder_27_layer_call_fn_2630443� ����������������C�@
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
8__inference_transformer_encoder_27_layer_call_fn_2630482� ����������������C�@
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
S__inference_transformer_encoder_28_layer_call_and_return_conditional_losses_2631110� ����������������C�@
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
S__inference_transformer_encoder_28_layer_call_and_return_conditional_losses_2631284� ����������������C�@
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
8__inference_transformer_encoder_28_layer_call_fn_2630883� ����������������C�@
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
8__inference_transformer_encoder_28_layer_call_fn_2630922� ����������������C�@
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
S__inference_transformer_encoder_29_layer_call_and_return_conditional_losses_2631550� ����������������C�@
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
S__inference_transformer_encoder_29_layer_call_and_return_conditional_losses_2631724� ����������������C�@
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
8__inference_transformer_encoder_29_layer_call_fn_2631323� ����������������C�@
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
8__inference_transformer_encoder_29_layer_call_fn_2631362� ����������������C�@
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