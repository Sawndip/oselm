%CLASS_INTERFACE Example MATLAB class wrapper to an underlying C++ class
classdef oselm < handle
    properties (SetAccess = private, Hidden = true)
        objectHandle; % Handle to the underlying C++ class instance
        isTrained = false;
    end
    methods
        %% Constructor - Create a new C++ class instance 
        function this = oselm(numNeuron, varargin)
            this.objectHandle = oselm_mex('new', numNeuron, varargin{:});
        end
        
        %% Destructor - Destroy the C++ class instance
        function delete(this)
            oselm_mex('delete', this.objectHandle);
        end

        %% init_train: Initial stage of training
        function init_train(this, xTrain, yTrain)
            oselm_mex('init_train', this.objectHandle, xTrain, yTrain);
            this.isTrained = true;
        end
        
        %% update: Update the oselm
        function update(this, xTrainNew, yTrainNew)
            assert(this.isTrained);
            oselm_mex('update', this.objectHandle, xTrainNew, yTrainNew);
        end
        
        %% compute_score: compute score given samples
        function varargout = compute_score(this, xTrain, varargin)
            assert(this.isTrained);
            [varargout{1:nargout}] = oselm_mex('compute_score', this.objectHandle, xTrain);
            % rescale to probabilty distribution (sum each row to 1)
            % The last term determines whether normalization should be performed.
            normalized = true;
            if nargin > 2, normalized = varargin{1}; end
            if nargout > 0 && normalized
                varargout{1} = bsxfun(@rdivide, exp(varargout{1}), sum(exp(varargout{1}), 2));
                varargout{1}(isnan(varargout{1})) = 0;
            end
        end
        
        %% snapshot: save current state for further use
        function snapshot(this, filename)
            oselm_mex('snapshot', this.objectHandle, filename);
        end
        
        %% load_snapshot: load a saved snapshot
        function load_snapshot(this, filename)
            oselm_mex('load_snapshot', this.objectHandle, filename);
        end

        %% Train: if not trained use `init_train`, else use `update`
        % This is reasonable since we use regularized ELM and num_samples < num_neuron is allowed.
        function train(this, xTrain, yTrain)
            if this.isTrained
                update(this, xTrain, yTrain);
            else
                init_train(this, xTrain, yTrain);
            end
            this.isTrained = true;
        end
        %% Test
        function varargout = test(this, xTest, yTest)
            assert(this.isTrained);
            [varargout{1:nargout}] = oselm_mex('test', this.objectHandle, xTest, yTest);
        end
    end
end