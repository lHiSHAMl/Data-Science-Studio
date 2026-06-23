from models.scripts.Regression.model_script import model_script as Reg_model_script ,validate_model as  Reg_validate_model
from models.scripts.Regression.model_report import model_report as Reg_model_report
from models.scripts.Regression.model_config import model_config as Reg_model_config,model_description as  Reg_model_description , model_reference_code as Reg_model_reference_code
# Import 
from models.scripts.hierarchical_clustering.model_config import hierarchy_config, hierarchy_model_description , hierarchy_model_reference_code
from models.scripts.hierarchical_clustering.model_report import model_report as hierarchy_report
from models.scripts.hierarchical_clustering.model_script import hierarchy_script,hierarchy_validate_model

# Import KNN functions
from models.scripts.classification.Knn.model_script import model_script as knn_script, validate_model as knn_validate
from models.scripts.classification.Knn.model_report import model_report as knn_report
from models.scripts.classification.Knn.model_config import model_config as knn_config, KNN_model_description , KNN_model_reference_code
# from Stream.scripts.reporting_lib.model_reporting_setup import display_ml_reporting_ui as knn_display_ml_reporting_dialogue

# ****************************add your models imports here as that**********************************************************************
# Add this import at the top
from models.scripts.classification.logistic_regression.model_script import model_script as lr_script, validate_model as lr_validate
from models.scripts.classification.logistic_regression.model_report import model_report as lr_report
from models.scripts.classification.logistic_regression.model_config import model_config as lr_config ,LR_model_description , LR_model_reference_code
# *********************************************************************************************************************

# Import KMeans functions
from models.scripts.k_means.model_script import model_script as Kmeans_model_script ,validate_model as Kmeans_validate_model
from models.scripts.k_means.model_report import model_report as Kmeans_model_report
from models.scripts.k_means.Model_config import model_config as Kmeans_model_config,model_description as Kmeans_model_description, model_reference_code as Kmeans_model_reference_code
# **********************************************************************************************
# Import new models - CORRECTED FOR YOUR STRUCTURE
from models.scripts.classification.Random_Forest_classifer.model_script import model_script as rf_script, validate_model as rf_validate
from models.scripts.classification.Random_Forest_classifer.model_report import model_report as rf_report
from models.scripts.classification.Random_Forest_classifer.model_config import model_config as rf_config, model_description as RF_model_description, model_reference_code as RF_model_reference_code

from models.scripts.classification.Gradient_boosting_classifer.model_script import model_script as gb_script, validate_model as gb_validate
from models.scripts.classification.Gradient_boosting_classifer.model_report import model_report as gb_report
from models.scripts.classification.Gradient_boosting_classifer.model_config import model_config as gb_config, model_description as GB_model_description, model_reference_code as GB_model_reference_code

from models.scripts.unsupervised_knn.model_script import model_script as unsup_knn_script, validate_model as unsup_knn_validate
from models.scripts.unsupervised_knn.model_report import model_report as unsup_knn_report
from models.scripts.unsupervised_knn.model_config import model_config as unsup_knn_config, model_description as UnsupKNN_model_description, model_reference_code as UnsupKNN_model_reference_code
# *********************************************************************************************************************
from models.scripts.ensemble.gradient_boosting_regressor.model_script import model_script as gbr_script, validate_model as gbr_validate
from models.scripts.ensemble.gradient_boosting_regressor.model_report import model_report as gbr_report
from models.scripts.ensemble.gradient_boosting_regressor.model_config import model_config as gbr_config, GBR_model_description, GBR_model_reference_code
# *********************************************************************************************************************
from models.scripts.DBSCAN.model_script import model_script as dbscan_script, validate_model as dbscan_validate
from models.scripts.DBSCAN.model_report import model_report as dbscan_report
from models.scripts.DBSCAN.model_config import model_config as dbscan_config, DBSCAN_model_description, DBSCAN_model_reference_code
# *********************************************************************************************************************
from models.scripts.classification.Naive_Bayes.model_script import model_script as NB_script, validate_model as NB_validate
from models.scripts.classification.Naive_Bayes.model_report import model_report as NB_report
from models.scripts.classification.Naive_Bayes.model_config import model_config as NB_config, model_description as NB_model_description , model_reference_code as NB_model_reference_code
# *********************************************************************************************************************
from models.scripts.classification.SVC.model_script import model_script as SVC_script, validate_model as SVC_validate
from models.scripts.classification.SVC.model_report import model_report as SVC_report
from models.scripts.classification.SVC.model_config import model_config as SVC_config, model_description as SVC_model_description , model_reference_code as SVC_model_reference_code
#*********************************************************************************************************************************
from models.scripts.classification.Decision_Tree.model_script import model_script as decisionTreeClassifier_script, validate_model as decisionTreeClassifier_validate
from models.scripts.classification.Decision_Tree.model_report import model_report as decisionTreeClassifier_report 
from models.scripts.classification.Decision_Tree.model_config import model_config as decisionTreeClassifier_config, DT_model_reference_code as DT_Refrence,DT_model_description as DT_Description
# *********************************************************************************************************************
from models.scripts.Knn_Regressor.model_script import model_script as KNN_Regressor_script, validate_model as KNN_Regressor_validate
from models.scripts.Knn_Regressor.model_report import model_report as KNN_Regressor_report
from models.scripts.Knn_Regressor.model_config import model_config as KNN_Regressor_config,KNN_model_reference_code as KNN_Regressor_Refrence,KNN_model_description as Knn_Regressor_Description
# *********************************************************************************************************************
from models.scripts.Random_Forest_Regressor.model_script import model_script as RF_Reg_model_script ,validate_model as RF_Reg_validate_model
from models.scripts.Random_Forest_Regressor.model_report import model_report as RF_Reg_model_report
from models.scripts.Random_Forest_Regressor.model_config import model_config as RF_Reg_model_config,model_description as RF_Reg_model_description , model_reference_code as RF_Reg_model_reference_code
#**********************************************************************************************************************
from models.scripts.Decision_Tree.model_script import model_script as dtr_script, validate_model as dtr_validate
from models.scripts.Decision_Tree.model_report import model_report as dtr_report
from models.scripts.Decision_Tree.model_config import model_config as dtr_config, model_description as dtr_model_description, model_reference_code as dtr_model_reference_code
#**********************************************************************************************************************

from models.scripts.SVR.model_script import model_script as svr_script, validate_model as svr_validate
from models.scripts.SVR.model_report import model_report as svr_report
from models.scripts.SVR.model_config import model_config as svr_config, model_description, model_reference_code
#**********************************************************************************************************************

from models.scripts.NN.model_script import model_script as nn_script, validate_model as nn_validate
from models.scripts.NN.model_report import model_report as nn_report
from models.scripts.NN.model_config import model_config as nn_config, model_description as NN_model_description, model_reference_code as NN_model_reference_code

def execute_model(model_name,action,param_dict):

    MODELS = {

        "Hierarchical Clustering": {
            "script": hierarchy_script,
            "report": hierarchy_report,
            "config":hierarchy_config,
            "validate":hierarchy_validate_model,
            "script_params": ["df", "features", "n_clusters", "linkage", "metric", "compute_full_tree", "distance_threshold", "edit"],
            "config_params":["model_data","edit"],
            "validate_params":["params"],

        },
             "Random Forest Regression": {
            "script": RF_Reg_model_script,
            "report": RF_Reg_model_report,
            "config":RF_Reg_model_config,
            "validate":RF_Reg_validate_model,
            "script_params": ["df", "features", "target", "edit", "use_grid_search", "param_grid", "manual_params", "cv_folds"],
            "config_params":["model_data","edit"],
            "validate_params":["params"],

        },
        "KNN": {
        "script": knn_script,
        "report": knn_report,
        "config": knn_config,
        "validate": knn_validate,
        # "prepare_report":knn_display_ml_reporting_dialogue,
        "script_params": ["df", "features", "target", "edit", "use_grid_search", "param_grid", "manual_params", "cv_folds"],
        "config_params": ["model_data", "edit"],
        "validate_params": ["params"],
        # "prepare_report_params":["model_name","model_index"]

    },
    # ************************************add your models here as that**********************************************************************
    "Logistic Regression Classifier": {
        "script": lr_script,
        "report": lr_report,
        "config": lr_config,
        "validate": lr_validate,
        "script_params": ["df", "features", "target", "edit", "use_grid_search", "param_grid", "manual_params", "cv_folds", "solver", "max_iter"],
        "config_params": ["model_data", "edit"],
        "validate_params": ["params"],

    },
    "Decision Tree Regressor": {
        "script": dtr_script,
        "report": dtr_report,
        "config": dtr_config,
        "validate": dtr_validate,
        "script_params": ["df", "features", "target", "edit", "use_grid_search", "param_grid", "manual_params", "cv_folds"],
        "config_params": ["model_data", "edit"],
        "validate_params": ["params"]
    },
    "KMeans Clustering": {
        "script": Kmeans_model_script,
        "report": Kmeans_model_report,
        "config": Kmeans_model_config,
        "validate": Kmeans_validate_model,
        "script_params": ["df", "features", "k_method", "auto_k", "manual_k", "max_k", "edit"],
        "config_params": ["model_data", "edit"],
        "validate_params": ["params"],
    },
    "Linear Regression": {
        "script": Reg_model_script,
        "report": Reg_model_report,
        "config": Reg_model_config,
        "validate": Reg_validate_model,
        "script_params": ["df", "features", "target", "edit", "model_type", "use_grid_search", "param_grid", "manual_params", "cv_folds"],
        "config_params": ["model_data", "edit"],
        "validate_params": ["params"],        
    },
     "Random Forest Classifier": {
        "script": rf_script,
        "report": rf_report,
        "config": rf_config,
        "validate": rf_validate,
        "script_params": ["df", "features", "target", "edit", "use_grid_search", "param_grid", "manual_params", "cv_folds"],
        "config_params": ["model_data", "edit"],
        "validate_params": ["params"]
    },
    "Gradient Boosting Classifier": {
        "script": gb_script,
        "report": gb_report,
        "config": gb_config,
        "validate": gb_validate,
        "script_params": ["df", "features", "target", "edit", "use_grid_search", "param_grid", "manual_params", "cv_folds"],
        "config_params": ["model_data", "edit"],
        "validate_params": ["params"]
    },
    "Unsupervised KNN Clustering": {
        "script": unsup_knn_script,
        "report": unsup_knn_report,
        "config": unsup_knn_config,
        "validate": unsup_knn_validate,
        "script_params": ["df", "features", "edit", "use_grid_search", "param_grid", "manual_params", "n_clusters"],
        "config_params": ["model_data", "edit"],
        "validate_params": ["params"]
    }
    ,
        "Gradient Boosting Regressor": {
    "script": gbr_script,
    "report": gbr_report,
    "config": gbr_config,
    "validate": gbr_validate,
    "script_params": ["df", "features", "target", "edit", "use_grid_search", "param_grid", "manual_params", "cv_folds"],
    "config_params": ["model_data", "edit"],
    "validate_params": ["params"],
},
    "DBSCAN": {
    "script": dbscan_script,
    "report": dbscan_report,
    "config": dbscan_config,
    "validate": dbscan_validate,
    "script_params": ["df", "features", "edit", "use_grid_search", "param_grid", "manual_params", "cv_folds"],
    "config_params": ["model_data", "edit"],
    "validate_params": ["params"],
},
    "Naive Bayes Classifier": {
        "script": NB_script,
        "report": NB_report,
        "config": NB_config,
        "validate": NB_validate,
        "script_params": ["df", "features", "target", "edit", "model_type", "use_grid_search", "param_grid", "manual_params", "cv_folds"],
        "config_params": ["model_data", "edit"],
        "validate_params": ["params"]
    },
    "Support Vector Machine Classifier": {
        "script": SVC_script,
        "report": SVC_report,
        "config": SVC_config,
        "validate": SVC_validate,
        "script_params": ["df", "features", "target", "edit", "use_grid_search", "param_grid", "manual_params", "cv_folds"],
        "config_params": ["model_data", "edit"],
        "validate_params": ["params"]
    },
    "Decision Tree Classifier": {
        "script":decisionTreeClassifier_script ,
        "report":decisionTreeClassifier_report ,
        "config": decisionTreeClassifier_config,
        "validate":decisionTreeClassifier_validate,
        "script_params": ["df", "features", "edit", "target", "use_grid_search","max_depth","min_samples_leaf","random_state"],
        "config_params": ["model_data", "edit"],
        "validate_params": ["params"]
    },
        "KNN_Regressor": {
        "script": KNN_Regressor_script,
        "report": KNN_Regressor_report,
        "config": KNN_Regressor_config,
        "validate": KNN_Regressor_validate,
        "script_params": ["df", "features", "target", "edit", "use_grid_search", "param_grid", "manual_params", "cv_folds"],
        "config_params": ["model_data", "edit"],
        "validate_params": ["params"]
    }
    ,
            "Support Vector Regression": {
                "script": svr_script,
                "report": svr_report,
                "config": svr_config,
                "validate": svr_validate,
                "script_params": ["df", "features", "target", "edit", "use_grid_search", "param_grid", "manual_params", "cv_folds"],
                "config_params": ["model_data", "edit"],
                "validate_params": ["params"]
            },

                    "Neural Network Classifier": {
            "script": nn_script,
            "report": nn_report,
            "config": nn_config,
            "validate": nn_validate,
            "script_params": ["df", "features", "target", "edit", "use_grid_search", "param_grid", "manual_params", "cv_folds"],
            "config_params": ["model_data", "edit"],
            "validate_params": ["params"]
        }
    }
    model_info = MODELS[model_name]
    
    # Pick function and expected params
    if action not in model_info:
        raise ValueError(f"Action '{action}' not found for model '{model_name}'")
    func = model_info[action]
    if action=="script":
        expected_params = model_info["script_params"]
    elif action =="config":
        expected_params = model_info['config_params']
        # st.write("config")
    elif action =="prepare_report":
        expected_params = model_info['prepare_report_params']
    elif action =="validate":
        expected_params = ["params"]
        # model_info['validate_params']
        # st.write("validate")
    elif action == "model_description":
        fun = None
        return  model_info['model_description']
    elif action == "model_reference":
        fun = None
        return model_info['model_reference']

    else : expected_params ={}
    # st.write(param_dict)
    # Extract only the needed params for this model
    args = {k: param_dict[k] for k in expected_params if k in param_dict}

    # st.write(args)
    # Call the model function dynamically
    return func(**args)

def get_model_data(model_name,action):
    MODELS = {
        # "Simple Linear Regression": {
        #     "model_description":SLR_model_description,
        #     "model_reference":SLR_model_reference_code
        # },
        # "decision_tree": {
        #     "script": dt_script,
        #     "report": dt_report,
        #     "params": ["data", "features", "target", "max_depth", "min_samples"]
        # }
        "Hierarchical Clustering": {
            "model_description":hierarchy_model_description,
            "model_reference":  hierarchy_model_reference_code
        },
        "Random Forest Regression":{
            "model_description":RF_Reg_model_description,
            "model_reference":RF_Reg_model_reference_code
        },
        "KNN": {
        "model_description":KNN_model_description,
        "model_reference":KNN_model_reference_code
    },
    # ************************************add your models here as that**********************************************************************
    "Logistic Regression Classifier": {
        "model_description":LR_model_description,
        "model_reference":LR_model_reference_code
    },
    "KMeans Clustering": {
        "model_description":Kmeans_model_description,
        "model_reference":Kmeans_model_reference_code
    
    },
    "Linear Regression": {
        "model_description":Reg_model_description,
        "model_reference":Reg_model_reference_code
    },
            "Random Forest Classifier": {
            "model_description": RF_model_description,
            "model_reference": RF_model_reference_code
        },
        "Gradient Boosting Classifier": {
            "model_description": GB_model_description,
            "model_reference": GB_model_reference_code
        },
        "Unsupervised KNN Clustering": {
            "model_description": UnsupKNN_model_description,
            "model_reference": UnsupKNN_model_reference_code
        },
            "Gradient Boosting Regressor": {
    "model_description": GBR_model_description,
    "model_reference": GBR_model_reference_code
    },
    "DBSCAN": {
    "model_description": DBSCAN_model_description,
    "model_reference": DBSCAN_model_reference_code
    },
        "Naive Bayes Classifier": {
        "model_description": NB_model_description,
        "model_reference": NB_model_reference_code
    },
    "Support Vector Machine Classifier": {
        "model_description": SVC_model_description,
        "model_reference": SVC_model_reference_code
    },
     "Decision Tree Classifier": {
    "model_description": DT_Description,
    "model_reference": DT_Refrence
    },
    "KNN_Regressor": {
    "model_description": Knn_Regressor_Description,
    "model_reference": KNN_Regressor_Refrence
    },
        "Neural Network Classifier": {
    "model_description": NN_model_description,
    "model_reference": NN_model_reference_code
    }
    }
     # Pick function and expected params
    if model_name not in MODELS:
        return "No description available for this model."
    model_info = MODELS[model_name]
    
    if action == "model_description":
        return  model_info['model_description']
    elif action == "model_reference":
        return model_info['model_reference']