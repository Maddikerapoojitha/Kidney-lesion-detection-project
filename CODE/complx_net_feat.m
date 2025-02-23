function out1 = complx_net_feat(varargin)
persistent INFO;
if (nargin < 1), error(message('nnet:Args:NotEnough')); end
in1 = varargin{1};
if ischar(in1)
  switch in1
    case 'info',
      if isempty(INFO), INFO = get_info; end
      out1 = INFO;
  end
else
  out1 = create_network(varargin{:});
end

function info = get_info

info.function = mfilename;
info.name = 'DCNN network ';
info.description = nnfcn.get_mhelp_title(mfilename);
info.type = 'nntype.network_fcn';
info.version = 6.0;

%%
function net = create_network(varargin)

if nargin < 2, error(message('nnet:Args:NotEnough')), end

v1 = varargin{1};
if isa(v1,'cell'), v1 = cell2mat(v1); end
v2 = varargin{2};
if nargin > 2, v3 = varargin{3}; end

if (nargin<= 6) && (size(v1,2)==2) && (~iscell(v2)) && (size(v2,1)==1) && ((nargin<3)||iscell(v3))
  nnerr.obs_use(mfilename,['See help for ' upper(mfilename) ' to update calls to the new argument list.']);
  net = new_5p0(varargin{:});
else
  net = new_5p1(varargin{:});
end

%=============================================================
function net = new_5p1(p,t,s,tf,btf,blf,pf,ipf,tpf,ddf)

if nargin < 2, error(message('nnet:Args:NotEnough')), end

% Defaults
if (nargin < 3), s = []; end
if (nargin < 4), tf = {}; end
if (nargin < 5), btf = 'trainlm'; end
if (nargin < 6), blf = 'learngdm'; end
if (nargin < 7), pf = 'mse'; end
if (nargin < 8), ipf = {'fixunknowns','removeconstantrows','mapminmax'}; end
if (nargin < 9), tpf = {'removeconstantrows','mapminmax'}; end
if (nargin < 10), ddf = 'dividerand'; end

% Format
if isa(p,'cell'), p = cell2mat(p); end
if isa(t,'cell'), t = cell2mat(t); end

% Error checking
if ~(isa(p,'double') || isreal(p)  || islogical(t))
  error(message('nnet:NNet:XNotLegal'))
end
if ~(isa(t,'double') || isreal(t) || islogical(t))
  error(message('nnet:NNet:TNotLegal'))
end
if isa(s,'cell')
  if (size(s,1) ~= 1)
    error(message('nnet:NNet:LayerSizes'))
  end
  for i=1:length(s)
    si = s{i};
    if ~isa(si,'double') || ~isreal(si) || any(size(si) ~= 1) || any(si<1) || any(round(si) ~= si)
      error(message('nnet:NNet:LayerSizes'))
    end
  end
  s = cell2mat(s);
end
if (~isa(s,'double')) || ~isreal(s) || (size(s,1) > 1) || any(s<1) || any(round(s) ~= s)
  error(message('nnet:NNet:LayerSizes'))
end

% Architecture
Nl = length(s)+1;
net = network;
net.numInputs = 1;
net.numLayers = Nl;
net.biasConnect = ones(Nl,1);
net.inputConnect(1,1) = 1;
[j,i] = meshgrid(1:Nl,1:Nl);
net.layerConnect = (j == (i-1));
net.outputConnect(Nl) = 1;

% Simulation
net.inputs{1}.processFcns = ipf;
for i=1:Nl
  if (i < Nl)
    net.layers{i}.size = s(i);
    if (Nl == 2)
      net.layers{i}.name = 'Hidden Layer';
    else
      net.layers{i}.name = ['Hidden Layer ' num2str(i)];
    end
  else
    net.layers{i}.name = 'Output Layer';
  end
  if (length(tf) < i) || all(isnan(tf{i}))
    if (i<Nl)
      net.layers{i}.transferFcn = 'tansig';
    else
      net.layers{i}.transferFcn = 'purelin';
    end
  else
    net.layers{i}.transferFcn = tf{i};
  end
end
net.outputs{Nl}.processFcns = tpf;

% Adaption
net.adaptfcn = 'adaptwb';
net.inputWeights{1,1}.learnFcn = blf;
for i=1:Nl
  net.biases{i}.learnFcn = blf;
  net.layerWeights{i,:}.learnFcn = blf;
end

% Training
net.trainfcn = btf;
net.dividefcn = ddf;
net.performFcn = pf;

% Initialization
net.initFcn = 'initlay';
for i=1:Nl
  net.layers{i}.initFcn = 'initnw';
end

% Configuration
% Warning: Use of these properties is no longer recommended
net.inputs{1}.exampleInput = p;
net.outputs{Nl}.exampleOutput = t;

% Initialize
net = init(net);

% Plots
net.plotFcns = {'plotperform','plottrainstate','plotregression'};

%================================================================
function net = new_5p0(p,s,tf,btf,blf,pf)
% Backward compatible to NNT 5.0

if nargin < 2, error(message('nnet:Args:NotEnough')), end

% Defaults
Nl = length(s);
if nargin < 3, tf = {'tansig'}; tf = tf(ones(1,Nl)); end
if nargin < 4, btf = 'trainlm'; end
if nargin < 5, blf = 'learngdm'; end
if nargin < 6, pf = 'mse'; end

% Error checking
if isa(p,'cell') && all(size(p)==[1 1]), p = p{1,1}; end
if (~isa(p,'double')) || ~isreal(p)
  error(message('nnet:NNData:XNotMatorCell1Mat'))
end
if isa(s,'cell')
  if (size(s,1) ~= 1)
    error(message('nnet:NNet:LayerSizes'))
  end
  for i=1:length(s)
    si = s{i};
    if ~isa(si,'double') || ~isreal(si) || any(size(si) ~= 1) || any(si<1) || any(round(si) ~= si)
      error(message('nnet:NNet:LayerSizes'))
    end
  end
  s = cell2mat(s);
end
if (~isa(s,'double')) || ~isreal(s) || (size(s,1) ~= 1) || any(s<1) || any(round(s) ~= s)
  error(message('nnet:NNet:LayerSizes'))
end

% Architecture
net = network(1,Nl);
net.biasConnect = ones(Nl,1);
net.inputConnect(1,1) = 1;
[j,i] = meshgrid(1:Nl,1:Nl);
net.layerConnect = (j == (i-1));
net.outputConnect(Nl) = 1;

% Simulation
for i=1:Nl
  net.layers{i}.size = s(i);
  net.layers{i}.transferFcn = tf{i};
end

% Performance
net.performFcn = pf;

% Adaption
net.adaptfcn = 'adaptwb';
net.inputWeights{1,1}.learnFcn = blf;
for i=1:Nl
  net.biases{i}.learnFcn = blf;
  net.layerWeights{i,:}.learnFcn = blf;
end

% Training
net.trainfcn = btf;

% Initialization
net.initFcn = 'initlay';
for i=1:Nl
  net.layers{i}.initFcn = 'initnw';
end

% Warning: this property is no longer recommended for use
net.inputs{1}.exampleInput = p;

net = init(net);

% Plots
net.plotFcns = {'plotperform','plottrainstate','plotregression'};
