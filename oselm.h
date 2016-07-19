#ifndef __OSELM_H_
#define __OSELM_H_

#include "elm_base.h"

template <typename dataT, bool isColMajor = true>
class oselm : public elm_base<dataT, isColMajor>
{
public:
	using matrixT = typename elm_base<dataT, isColMajor>::matrixT;
	using matrixMapT = typename elm_base<dataT, isColMajor>::matrixMapT;

	explicit oselm(int num_neuron, dataT regularity_const = 0, ostream &os = std::cout)
		: elm_base<dataT, isColMajor>(num_neuron, regularity_const, os)
	{}	// by default regularity_const is set to zero

	virtual ~oselm() {}

	// Update that is central to oselm
	// Note that run-time sanity check for dimensions of data is not available.
	// This is because m_featureLength and m_numClass are known in oselm_init_training.
	// So the user is responsible for ensuring xTrain_new and yTrain_new has the required memory
	// They will be wrapped into matrix of size  {batch_size, m_featureLength} and {batch_size, m_numClass} respectively.
	int update(dataT *xTrain_new, dataT *yTrain_new, int batch_size)
	{
		this->m_os << "--Update on oselm begins.--\n";
		this->tic();
		matrixMapT xTrain = this->wrap_data(xTrain_new, batch_size, this->m_featureLength);
		matrixMapT yTrain = this->wrap_data(yTrain_new, batch_size, this->m_numClass);
		matrixT H, lhs, rhs, sol;
		H = this->compute_H_matrix(xTrain);
		rhs = H * m_P;
		lhs = rhs * H.transpose() + matrixT::Identity(H.rows(), H.rows());
		auto isSolved = solve_eigen(sol, lhs, rhs);
		elm_assert(isSolved);
		m_P = m_P - rhs.transpose() * sol;
		this->m_beta = this->m_beta + m_P * H.transpose() * (yTrain - H * this->m_beta);
		this->toc();
		this->m_os << "--Update finishes.--\n";
		return 0;
	}
	// This is a wrapper of elm_train for unifying naming.
	int oselm_init_train(dataT *xTrain, int xRows, int xCols,
		dataT *yTrain, int yRows, int yCols)
	{
		return elm_train(xTrain, xRows, xCols, yTrain, yRows, yCols);
	}
	// Reimplement elm_train to calculate the P matrix needed to update in the oselm process.
	virtual int elm_train(dataT *xTrainPtr, int xRows, int xCols,
		dataT *yTrainPtr, int yRows, int yCols) override
	{
		this->m_os << "--Training begins.--\n";
		this->tic();
		if (abs(this->m_regConst) < 1e-7)  // if not regulairized, inforce the condition to prevent H.t()*H from degrading
		{
			elm_assert(xRows >= this->m_numNeuron);
		}	
		elm_assert(xRows == yRows);
		this->m_featureLength = xCols;
		this->m_numClass = yCols;
		this->m_weight = matrixT(this->m_numNeuron, this->m_featureLength);
		matrixMapT xTrain = this->wrap_data(xTrainPtr, xRows, xCols);
		matrixMapT yTrain = this->wrap_data(yTrainPtr, yRows, yCols);
		random_init(this->m_weight.data(), (int)this->m_weight.size(), this->m_range);
		matrixT H = this->compute_H_matrix(xTrain);
		matrixT lhs = H.transpose() * H + matrixT::Identity(this->m_numNeuron, this->m_numNeuron) * this->m_regConst;
		matrixT rhs = H.transpose() * yTrain;
		auto isSolved = solve_eigen(this->m_beta, lhs, rhs);
		elm_assert(isSolved);
		matrixT P_rhs = matrixT::Identity(this->m_numNeuron, this->m_numNeuron);
		auto isSolvedP = solve_eigen(m_P, lhs, P_rhs);
		elm_assert(isSolvedP);
		this->toc();
		this->m_os << "--Training is finished.--\n";
		return 0;
	}

	vector<dataT> oselm_test(dataT *xTestPtr, int xRows, int xCols,
		dataT *yTestPtr, int yRows, int yCols,
		dataT threshold = 0)
	{
		return this->elm_test(xTestPtr, xRows, xCols, yTestPtr, yRows, yCols, threshold);
	}
	virtual int snapshot(const string &filename) override
	{
		auto flag = elm_base<dataT, isColMajor>::snapshot(filename);
		elm_assert(flag == 0);
		fstream out(filename, std::ios::out | std::ios::app | std::ios::binary);
		elm_assert(out.is_open());
		serialize(this->m_P, out, "P");
		out.close();
		return 0;
		// return serialize(this->m_P, filename, "P");
	}
	// Unlike snapshot, there is no way to find the position of each variable,
	// I have to reimplement this function from scratch.
	// TODO: Refactor for a cleaner realization 
	virtual int load_snapshot(const string &filename) override
	{
		fstream in(filename, std::ios::in | std::ios::binary);
		if (!in.is_open())
		{
			this->m_os << "Cannot load snapshot " << filename << "\n";
			return 1;
		}
		deserialize(this->m_weight, in, "weight");
		deserialize(this->m_beta, in, "beta");
		deserialize(this->m_numNeuron, in, "numNeuron");
		deserialize(this->m_featureLength, in, "featureLength");
		deserialize(this->m_regConst, in, "regConst");
		deserialize(this->m_range, in, "range");
		deserialize(this->m_numClass, in, "numClasses");
		deserialize(this->m_P, in, "P");
		in.close();
		return 0;
	}

protected:
	matrixT m_P;	// The only matrix that is needed to store.  See paper for details.
};

#endif // __OSELM_H__