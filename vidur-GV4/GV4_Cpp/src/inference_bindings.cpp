#include "gv4/inference.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace gv4 {
namespace {

struct PythonModels {
    py::object controller_value;
    py::object adversary_value;
    py::object controller_policy;
    py::object adversary_policy;
};

void check_model(
    const py::object& model,
    const char* role,
    const char* label,
    const Config& config) {
    if (model.is_none()) {
        return;
    }
    const std::string actual_role = py::hasattr(model, "role")
        ? py::str(model.attr("role")).cast<std::string>()
        : role;
    if (actual_role != role) {
        throw std::invalid_argument(std::string(label) + " model has the wrong role");
    }
    if (!py::hasattr(model, "feature_schema_version") ||
        py::str(model.attr("feature_schema_version")).cast<std::string>() !=
            config.feature_schema_version) {
        throw std::invalid_argument(std::string(label) + " model schema mismatch");
    }
    if (!py::hasattr(model, "config_manifest_sha256") ||
        py::str(model.attr("config_manifest_sha256")).cast<std::string>() !=
            config.manifest_sha256) {
        throw std::invalid_argument(std::string(label) + " model manifest mismatch");
    }
}

std::vector<float> model_vector(
    py::object value,
    std::size_t expected,
    const char* label) {
    if (py::hasattr(value, "detach")) {
        value = value.attr("detach")();
    }
    if (py::hasattr(value, "cpu")) {
        value = value.attr("cpu")();
    }
    if (py::hasattr(value, "numpy")) {
        value = value.attr("numpy")();
    }
    auto array = py::array_t<float, py::array::c_style | py::array::forcecast>::ensure(value);
    if (!array || static_cast<std::size_t>(array.size()) != expected) {
        throw std::runtime_error(std::string(label) + " returned the wrong shape");
    }
    const float* data = array.data();
    std::vector<float> result(data, data + array.size());
    for (const float item : result) {
        if (!std::isfinite(item)) {
            throw std::runtime_error(std::string(label) + " returned a non-finite value");
        }
    }
    return result;
}

std::unique_ptr<InferenceRuntime> make_inference(
    Config config,
    py::object controller_value,
    py::object adversary_value,
    py::object controller_policy,
    py::object adversary_policy) {
    check_model(controller_value, "controller", "value", config);
    check_model(adversary_value, "adversary", "value", config);
    check_model(controller_policy, "controller", "policy", config);
    check_model(adversary_policy, "adversary", "policy", config);
    auto models = std::make_shared<PythonModels>(PythonModels{
        std::move(controller_value),
        std::move(adversary_value),
        std::move(controller_policy),
        std::move(adversary_policy)});

    ValuePredictor value = [models](
        Player player,
        const std::vector<StateFeatures>& states) {
        py::gil_scoped_acquire acquire;
        const py::object& model = player == Player::Controller
            ? models->controller_value
            : models->adversary_value;
        if (model.is_none()) {
            throw std::runtime_error("no value model is configured for this role");
        }
        py::object result = model.attr("predict_structured")(states);
        return model_vector(result, states.size(), "value model");
    };
    ControllerPolicyPredictor controller = [models](
        const StateFeatures& state,
        const std::vector<ControllerActionFeatures>& actions) {
        py::gil_scoped_acquire acquire;
        if (models->controller_policy.is_none()) {
            throw std::runtime_error("no controller policy model is configured");
        }
        py::object result = models->controller_policy.attr("predict_root_structured")(
            state, actions);
        return model_vector(result, actions.size(), "controller policy model");
    };
    AdversaryPolicyPredictor adversary = [models](
        const StateFeatures& state,
        const std::vector<AdversaryActionFeatures>& actions) {
        py::gil_scoped_acquire acquire;
        if (models->adversary_policy.is_none()) {
            throw std::runtime_error("no adversary policy model is configured");
        }
        py::object result = models->adversary_policy.attr("predict_root_structured")(
            state, actions);
        return model_vector(result, actions.size(), "adversary policy model");
    };
    return std::make_unique<InferenceRuntime>(
        std::move(config),
        std::move(value),
        std::move(controller),
        std::move(adversary));
}

}  // namespace

void bind_inference_types(py::module_& module) {
    py::class_<InferenceRuntime>(module, "InferenceRuntime")
        .def(
            py::init(&make_inference),
            py::arg("config"),
            py::arg("controller_value_model") = py::none(),
            py::arg("adversary_value_model") = py::none(),
            py::arg("controller_policy_model") = py::none(),
            py::arg("adversary_policy_model") = py::none())
        .def_property_readonly(
            "config",
            &InferenceRuntime::config,
            py::return_value_policy::reference_internal)
        .def_property_readonly(
            "builder",
            &InferenceRuntime::builder,
            py::return_value_policy::reference_internal)
        .def("build_state_features", &InferenceRuntime::build_state_features)
        .def("predict_values", &InferenceRuntime::predict_values)
        .def("predict_value", &InferenceRuntime::predict_value)
        .def("predict_controller_logits", &InferenceRuntime::predict_controller_logits)
        .def("predict_adversary_logits", &InferenceRuntime::predict_adversary_logits);
}

}  // namespace gv4
