%CLASS_INTERFACE Example MATLAB class wrapper to an underlying C++ class
classdef oselm < handle
    properties (SetAccess = private, Hidden = true)
        objectHandle; % Handle to the underlying C++ class instance
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
        end
        
        %% update: Update the oselm
        function update(this, xTrainNew, yTrainNew)
            oselm_mex('update', this.objectHandle, xTrainNew, yTrainNew);
        end
        
        %% compute_score: compute score given samples
        function varargout = compute_score(this, xTrain)
            [varargout{1:nargout}] = oselm_mex('compute_score', this.objectHandle, xTrain);
        end

        %% Test
        function varargout = test(this, xTest, yTest)
            [varargout{1:nargout}] = oselm_mex('test', this.objectHandle, xTest, yTest);
        end
    end
end