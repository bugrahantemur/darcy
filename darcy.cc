
#include <deal.II/base/function.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/tensor_function.h>
#include <deal.II/base/timer.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>
#include <stdlib.h>

#include <fstream>
#include <iostream>
#include <memory>

namespace Darcy {
template <int dim>
class DarcyProblem {
 public:
  DarcyProblem(dealii::ParameterHandler &parameter_handler,
               unsigned int const degree);
  void run();

 private:
  void make_grid();
  void setup_dofs();
  void assemble_system();
  void solve();
  void output_results() const;

  dealii::ParameterHandler &prm;

  unsigned int const degree;

  dealii::Triangulation<dim> triangulation;
  dealii::FESystem<dim> fe;
  dealii::DoFHandler<dim> dof_handler;

  dealii::AffineConstraints<double> constraints;

  dealii::BlockSparsityPattern sparsity_pattern;
  dealii::BlockSparseMatrix<double> system_matrix;

  dealii::BlockVector<double> solution;
  dealii::BlockVector<double> system_rhs;
};

template <int dim>
class InflowDirichletBoundaryValues : public dealii::Function<dim> {
 public:
  InflowDirichletBoundaryValues() : dealii::Function<dim>(dim + 1) {}

  double value(dealii::Point<dim> const &p,
               unsigned int const component = 0) const override;

  void vector_value(dealii::Point<dim> const &p,
                    dealii::Vector<double> &value) const override;
};

template <int dim>
double InflowDirichletBoundaryValues<dim>::value(
    dealii::Point<dim> const &p, unsigned int const component) const {
  (void)p;

  Assert(component < this->n_components,
         dealii::ExcIndexRange(component, 0, this->n_components));

  return (component == 0) ? 1. : 0.;
}

template <int dim>
void InflowDirichletBoundaryValues<dim>::vector_value(
    dealii::Point<dim> const &p, dealii::Vector<double> &values) const {
  for (unsigned int c = 0; c < this->n_components; ++c)
    values(c) = InflowDirichletBoundaryValues<dim>::value(p, c);
}

template <int dim>
class PressureNeumannBoundaryValues : public dealii::Function<dim> {
 public:
  PressureNeumannBoundaryValues() : dealii::Function<dim>() {}

  double value(dealii::Point<dim> const &p,
               unsigned int const component = 0) const override;
};

template <int dim>
double PressureNeumannBoundaryValues<dim>::value(
    dealii::Point<dim> const & /*p*/, unsigned int const /*component*/) const {
  return 0.;
}

template <int dim>
class RightHandSide : public dealii::TensorFunction<1, dim> {
 public:
  RightHandSide() : dealii::TensorFunction<1, dim>() {}

  dealii::Tensor<1, dim> value(dealii::Point<dim> const &p) const override;

  void value_list(std::vector<dealii::Point<dim>> const &p,
                  std::vector<dealii::Tensor<1, dim>> &value) const override;
};

template <int dim>
dealii::Tensor<1, dim> RightHandSide<dim>::value(
    dealii::Point<dim> const & /*p*/) const {
  return dealii::Tensor<1, dim>();
}

template <int dim>
void RightHandSide<dim>::value_list(
    std::vector<dealii::Point<dim>> const &vp,
    std::vector<dealii::Tensor<1, dim>> &values) const {
  for (unsigned int c = 0; c < vp.size(); ++c) {
    values[c] = RightHandSide<dim>::value(vp[c]);
  }
}

template <int dim>
class KInverse : public dealii::TensorFunction<2, dim> {
 public:
  KInverse() : dealii::TensorFunction<2, dim>() {}

  void value_list(std::vector<dealii::Point<dim>> const &points,
                  std::vector<dealii::Tensor<2, dim>> &values) const override;
};

template <int dim>
void KInverse<dim>::value_list(
    std::vector<dealii::Point<dim>> const &points,
    std::vector<dealii::Tensor<2, dim>> &values) const {
  (void)points;
  AssertDimension(points.size(), values.size());

  double const K_inv = 1. / 1.0e-7;

  for (auto &value : values)
    value = K_inv * dealii::unit_symmetric_tensor<dim>();
}

template <int dim>
DarcyProblem<dim>::DarcyProblem(dealii::ParameterHandler &parameter_handler,
                                unsigned int const degree)
    : prm(parameter_handler),
      degree(degree),
      fe(dealii::FE_Q<dim>(degree + 1), dim, dealii::FE_Q<dim>(degree), 1),
      dof_handler(triangulation) {}

class ParameterReader : public dealii::Subscriptor {
 public:
  ParameterReader(dealii::ParameterHandler &parameter_handler)
      : prm(parameter_handler) {}
  void read_parameters(std::string const &parameter_file);

 private:
  void declare_parameters();
  dealii::ParameterHandler &prm;
};

void ParameterReader::declare_parameters() {
  prm.enter_subsection("Mesh & geometry parameters");
  {
    prm.declare_entry(
        "Degree", "1", dealii::Patterns::Integer(1, 15),
        "Minimum polynomial degree in the element (polynomial degree varies "
        "between primary variables in Taylor-Hood elements)");
    prm.declare_entry("Number of refinements", "4",
                      dealii::Patterns::Integer(0),
                      "Number of global mesh refinement steps applied to the "
                      "initial course grid");
  }
  prm.leave_subsection();

  prm.enter_subsection("Physical constants");
  {
    prm.declare_entry("Dynamic viscosity", "0.0", dealii::Patterns::Double(0.),
                      "Dynamic viscosity (mu)");

    prm.declare_entry("Porosity", "0.0", dealii::Patterns::Double(0., 1.),
                      "Porosity (phi)");

    prm.declare_entry("Permeability", "0.0", dealii::Patterns::Double(0.),
                      "Permeability (k)");
  }
  prm.leave_subsection();

  prm.enter_subsection("Output parameters");
  {
    prm.declare_entry("Output filename", "solution",
                      dealii::Patterns::Anything(),
                      "Name of the output file (without extension)");
  }
  prm.leave_subsection();
}

void ParameterReader::read_parameters(std::string const &parameter_file) {
  declare_parameters();

  prm.parse_input(parameter_file);
}

template <int dim>
void DarcyProblem<dim>::make_grid() {
  dealii::GridGenerator::hyper_cube(triangulation);

  prm.enter_subsection("Mesh & geometry parameters");
  unsigned int const n_global_refinements =
      prm.get_integer("Number of refinements");
  prm.leave_subsection();

  triangulation.refine_global(n_global_refinements);

  // Boundary indicator
  for (auto const &cell : triangulation.active_cell_iterators())
    for (auto const &face : cell->face_iterators()) {
      if (std::abs(face->center()[0]) < 1.0e-12) face->set_all_boundary_ids(1);
      for (int d = 1; d < dim; ++d)
        if ((std::abs(face->center()[d]) < 1.0e-12) ||
            (std::abs(face->center()[d] - 1.) < 1.0e-12))
          face->set_all_boundary_ids(2);
    }
}

template <int dim>
void DarcyProblem<dim>::setup_dofs() {
  system_matrix.clear();
  dof_handler.distribute_dofs(fe);

  // velocity components are block 0 and pressure components are block 1
  auto const block_component = std::invoke([]() {
    std::vector<unsigned int> block_component(dim + 1, 0);
    block_component[dim] = 1;
    return block_component;
  });
  dealii::DoFRenumbering::component_wise(dof_handler, block_component);

  {
    constraints.clear();
    std::set<dealii::types::boundary_id> no_flux_boundaries{2};
    dealii::VectorTools::compute_no_normal_flux_constraints(
        dof_handler, 0, no_flux_boundaries, constraints);
  }

  constraints.close();

  std::vector<dealii::types::global_dof_index> const dofs_per_block =
      dealii::DoFTools::count_dofs_per_fe_block(dof_handler, block_component);

  dealii::types::global_dof_index const n_u = dofs_per_block[0];
  dealii::types::global_dof_index const n_p = dofs_per_block[1];

  std::cout << "  Number of active cells: " << triangulation.n_active_cells()
            << std::endl
            << "  Number of degrees of freedom: " << dof_handler.n_dofs()
            << " (" << n_u << "+" << n_p << ")" << std::endl;

  {
    dealii::BlockDynamicSparsityPattern dsp(dofs_per_block, dofs_per_block);

    auto const coupling = std::invoke([]() {
      dealii::Table<2, dealii::DoFTools::Coupling> coupling(dim + 1, dim + 1);

      for (unsigned int c = 0; c < dim + 1; ++c)
        for (unsigned int d = 0; d < dim + 1; ++d)
          if (!((c == dim) && (d == dim)))
            coupling[c][d] = dealii::DoFTools::always;
          else
            coupling[c][d] = dealii::DoFTools::none;

      return coupling;
    });

    dealii::DoFTools::make_sparsity_pattern(dof_handler, coupling, dsp,
                                            constraints, false);

    sparsity_pattern.copy_from(dsp);
  }

  system_matrix.reinit(sparsity_pattern);
  solution.reinit(dofs_per_block);
  system_rhs.reinit(dofs_per_block);
}

template <int dim>
void DarcyProblem<dim>::assemble_system() {
  system_matrix = 0.;
  system_rhs = 0.;

  dealii::QGauss<dim> quadrature_formula(degree + 2);
  dealii::QGauss<dim - 1> face_quadrature_formula(degree + 2);

  dealii::FEValues<dim> fe_values(
      fe, quadrature_formula,
      dealii::update_values | dealii::update_quadrature_points |
          dealii::update_JxW_values | dealii::update_gradients);

  dealii::FEFaceValues<dim> fe_face_values(
      fe, face_quadrature_formula,
      dealii::update_values | dealii::update_normal_vectors |
          dealii::update_quadrature_points | dealii::update_JxW_values);

  unsigned int const n_dofs_per_cell = fe.n_dofs_per_cell();
  unsigned int const n_q_points = quadrature_formula.size();
  unsigned int const n_face_q_points = face_quadrature_formula.size();

  dealii::FullMatrix<double> local_matrix(n_dofs_per_cell, n_dofs_per_cell);
  dealii::Vector<double> local_rhs(n_dofs_per_cell);

  std::vector<dealii::types::global_dof_index> local_dof_indices(
      n_dofs_per_cell);

  RightHandSide<dim> const right_hand_side;
  KInverse<dim> const k_inverse;
  PressureNeumannBoundaryValues<dim> pressure_boundary;

  std::vector<dealii::Tensor<1, dim>> rhs_values(n_q_points);
  std::vector<dealii::Tensor<2, dim>> k_inverse_values(n_q_points);
  std::vector<double> pressure_boundary_values(n_face_q_points);

  dealii::FEValuesExtractors::Vector const velocities(0);
  dealii::FEValuesExtractors::Scalar const pressure(dim);

  prm.enter_subsection("Physical constants");
  double const porosity = prm.get_double("Porosity");
  double const dyn_viscosity = prm.get_double("Dynamic viscosity");
  double const porosity_x_dyn_visc = porosity * dyn_viscosity;
  prm.leave_subsection();

  std::vector<double> div_phi_u(n_dofs_per_cell);
  std::vector<dealii::Tensor<1, dim>> phi_u(n_dofs_per_cell);
  std::vector<double> phi_p(n_dofs_per_cell);

  for (auto const &cell : dof_handler.active_cell_iterators()) {
    fe_values.reinit(cell);

    local_matrix = 0;
    local_rhs = 0;

    right_hand_side.value_list(fe_values.get_quadrature_points(), rhs_values);
    k_inverse.value_list(fe_values.get_quadrature_points(), k_inverse_values);

    // Domain integral
    for (unsigned int q = 0; q < n_q_points; ++q) {
      for (unsigned int k = 0; k < n_dofs_per_cell; ++k) {
        div_phi_u[k] = fe_values[velocities].divergence(k, q);
        phi_u[k] = fe_values[velocities].value(k, q);
        phi_p[k] = fe_values[pressure].value(k, q);
      }

      for (unsigned int i = 0; i < n_dofs_per_cell; ++i) {
        for (unsigned int j = 0; j < n_dofs_per_cell; ++j) {
          local_matrix(i, j) +=
              (phi_u[i] * porosity_x_dyn_visc * k_inverse_values[q] * phi_u[j] -
               div_phi_u[i] * phi_p[j] - phi_p[i] * div_phi_u[j]) *
              fe_values.JxW(q);
        }

        local_rhs(i) += phi_u[i] * rhs_values[q] * fe_values.JxW(q);
      }
    }

    // Boundary integral
    for (auto const &face : cell->face_iterators()) {
      if (face->boundary_id() == 0) {
        fe_face_values.reinit(cell, face);

        pressure_boundary.value_list(fe_face_values.get_quadrature_points(),
                                     pressure_boundary_values);

        for (unsigned int q = 0; q < n_face_q_points; ++q)
          for (unsigned int i = 0; i < n_dofs_per_cell; ++i)
            local_rhs(i) +=
                -(fe_face_values[velocities].value(i, q) *
                  fe_face_values.normal_vector(q) *
                  pressure_boundary_values[q] * fe_face_values.JxW(q));
      }
    }

    // Simultaneous assembly + Dirichlet conditions
    cell->get_dof_indices(local_dof_indices);
    constraints.distribute_local_to_global(
        local_matrix, local_rhs, local_dof_indices, system_matrix, system_rhs);
  }

  std::map<dealii::types::global_dof_index, double> boundary_values;
  dealii::VectorTools::interpolate_boundary_values(
      dof_handler, 1, InflowDirichletBoundaryValues<dim>(), boundary_values,
      fe.component_mask(velocities));
  dealii::MatrixTools::apply_boundary_values(boundary_values, system_matrix,
                                             solution, system_rhs);
}

template <int dim>
void DarcyProblem<dim>::solve() {
  dealii::SparseDirectUMFPACK A_direct;
  A_direct.initialize(system_matrix);
  A_direct.vmult(solution, system_rhs);
}

template <int dim>
void DarcyProblem<dim>::output_results() const {
  auto const solution_names = std::invoke([]() {
    std::vector<std::string> solution_names(dim, "velocity");
    solution_names.emplace_back("pressure");
    return solution_names;
  });

  std::vector<dealii::DataComponentInterpretation::DataComponentInterpretation>
      data_component_interpretation(
          dim,
          dealii::DataComponentInterpretation::component_is_part_of_vector);
  data_component_interpretation.emplace_back(
      dealii::DataComponentInterpretation::component_is_scalar);

  dealii::DataOut<dim> data_out;
  data_out.attach_dof_handler(dof_handler);
  data_out.add_data_vector(solution, solution_names,
                           dealii::DataOut<dim>::type_dof_data,
                           data_component_interpretation);
  data_out.build_patches();

  prm.enter_subsection("Output parameters");
  std::string const output_file_name = prm.get("Output filename");
  prm.leave_subsection();

  std::ofstream output(output_file_name + ".vtk");
  data_out.write_vtk(output);
}

template <int dim>
void DarcyProblem<dim>::run() {
  dealii::Timer timer;

  std::cout << "  Generating grid... "
            << "\n";
  timer.restart();
  make_grid();
  timer.stop();
  std::cout << "  ...done (" << timer.cpu_time() << " s)"
            << "\n\n";

  std::cout << "  Setting up DoFs... "
            << "\n";
  timer.restart();
  setup_dofs();
  timer.stop();
  std::cout << "  ...done (" << timer.cpu_time() << " s)"
            << "\n\n";

  std::cout << "  Assembling... "
            << "\n";
  timer.restart();
  assemble_system();
  timer.stop();
  std::cout << "  ...done (" << timer.cpu_time() << " s)"
            << "\n\n";

  std::cout << "  Solving... "
            << "\n";
  timer.restart();
  solve();
  timer.stop();
  std::cout << "  ...done (" << timer.cpu_time() << " s)"
            << "\n\n";

  output_results();

  std::cout << "\n";
}
}  // namespace Darcy

int main(int, char **) {
  try {
    dealii::ParameterHandler prm;
    Darcy::ParameterReader param(prm);
    param.read_parameters("input.json");

    Darcy::DarcyProblem<3> darcy_problem(prm, 2);
    darcy_problem.run();
  } catch (std::exception &exc) {
    std::cerr << std::endl
              << std::endl
              << "----------------------------------------------------"
              << std::endl;
    std::cerr << "Exception on processing: " << std::endl
              << exc.what() << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------"
              << std::endl;

    return EXIT_FAILURE;
  } catch (...) {
    std::cerr << std::endl
              << std::endl
              << "----------------------------------------------------"
              << std::endl;
    std::cerr << "Unknown exception!" << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------"
              << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
