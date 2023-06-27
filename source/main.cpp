#include <igl/eigs.h>
#include <igl/opengl/glfw/Viewer.h>
#include <igl/opengl/glfw/imgui/ImGuiPlugin.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <igl/opengl/glfw/imgui/ImGuiHelpers.h>
#include <igl/read_triangle_mesh.h>
#include <igl/massmatrix.h>
#include <iostream>
#include "trimesh.h"

Eigen::MatrixXd eigenvectors;
int selectedcolumn=0;

// This function is called every time a keyboard button is pressed
bool key_down(igl::opengl::glfw::Viewer& viewer, unsigned char key, int modifier)
{
  std::cout<<"Key: "<<key<<" "<<(unsigned int)key<<std::endl;
  if (key == '1')
  {
    selectedcolumn = 1;
    viewer.data().set_data(eigenvectors.col(selectedcolumn));
  }
  else if (key == '2')
  {
    selectedcolumn = 2;
    viewer.data().set_data(eigenvectors.col(selectedcolumn));
  }
  else if (key == ' ')
  {
    selectedcolumn = (selectedcolumn + 100) % eigenvectors.cols();
    viewer.data().set_data(eigenvectors.col(selectedcolumn));
  }

  return false;
}

void newslug(Eigen::SparseMatrix<double>& L,igl::opengl::glfw::Viewer& viewer,Eigen::MatrixXd& V,Eigen::MatrixXi& F)
{
  auto newV = V;
  viewer.data().set_mesh(newV,F);
  double lambda = 0.5; //if too big it will be too fast... conductivity is lambda
  Eigen::SparseMatrix<double> I(L.rows(), L.cols());
  I.setIdentity();
  Eigen::SparseMatrix<double> matrix = I - (-L * lambda);//SOLVE FOR X : newV * X = L*l*I 
  Eigen::SimplicialCholesky<Eigen::SparseMatrix<double>> llt; //see page 6 of paper
  llt.compute(matrix);
  // animation function
  viewer.callback_pre_draw = [&](igl::opengl::glfw::Viewer & )->bool
  {
    newV = llt.solve(newV); //here, newV becomes X
    viewer.data().set_mesh(newV,F);
    return false;
  };
  viewer.launch();
}

void heat(Eigen::SparseMatrix<double>& L,igl::opengl::glfw::Viewer& viewer)
{
  //auto init = eigenvectors.col(1);
  Eigen::MatrixXd init = Eigen::VectorXd{ L.rows() }.setRandom();
  double conductivity = 0.05; //need to be small, depends on material
  auto I = Eigen::MatrixXd{L.rows(), L.cols()}.setIdentity();
  auto x = init;
  Eigen::SparseMatrix<double> heatSmooth = I - (-L * conductivity);
  viewer.data().set_data(x);
  Eigen::SimplicialCholesky<Eigen::SparseMatrix<double>> llt;
  llt.compute(heatSmooth); //sparse matrix has no member llt? :(
  // animation function
  viewer.callback_pre_draw = [&](igl::opengl::glfw::Viewer & )->bool
  {
    x = llt.solve(x);
    viewer.data().set_data(x);
    return false;
  };
  viewer.launch();
}

void eigen(Eigen::SparseMatrix<double>& L,igl::opengl::glfw::Viewer& viewer, int eigen)
{
  selectedcolumn = eigen;
  viewer.data().set_data(eigenvectors.col(selectedcolumn));
  viewer.launch();
}

Eigen::SparseMatrix<double> getLaplacian(trimesh::trimesh_t& mesh, Eigen::MatrixXd& V)
{
    Eigen::SparseMatrix<double> Laplacian(V.rows(),V.rows());
    Laplacian.reserve(Eigen::VectorXi::Constant(V.rows(), 5));
    int LIndex = 0;
    for(int vi = 0; vi < V.rows(); ++vi) //iterate vertices
    {
      int w = vi;
      std::vector< trimesh::index_t > neighsOfw;
      mesh.vertex_vertex_neighbors( w, neighsOfw );

      for(int wi = 0; wi < neighsOfw.size(); ++wi) //iterate around w
      {
        //v~w condition
        int v = neighsOfw.at(wi);
        std::vector< trimesh::index_t > neighsOfv;
        mesh.vertex_vertex_neighbors( v, neighsOfv );
        std::vector< trimesh::index_t > common(6);
        std::sort(neighsOfw.begin(), neighsOfw.end());
        std::sort(neighsOfv.begin(), neighsOfv.end());
        std::set_intersection(neighsOfw.begin(), neighsOfw.end(),neighsOfv.begin(), neighsOfv.end(),common.begin());

        auto a = V.row(common.at(0)) - V.row(w);
        auto b = V.row(common.at(0)) - V.row(v);
        auto c = V.row(common.at(1)) - V.row(w);
        auto d = V.row(common.at(1)) - V.row(v);
        auto e = V.row(v) - V.row(w);
        double lv1 = sqrt(pow(a.x(),2) + pow(a.y(),2) + pow(a.z(),2));
        double lw1 = sqrt(pow(b.x(),2) + pow(b.y(),2) + pow(b.z(),2));
        double lv2 = sqrt(pow(c.x(),2) + pow(c.y(),2) + pow(c.z(),2));
        double lw2 = sqrt(pow(d.x(),2) + pow(d.y(),2) + pow(d.z(),2));
        double lvw = sqrt(pow(e.x(),2) + pow(e.y(),2) + pow(e.z(),2));
        double p1 = (lv1 + lw1 + lvw)/2;
        double p2 = (lv2 + lw2 + lvw)/2;
        double T1 = sqrt(p1*(p1-lv1)*(p1-lw1)*(p1-lvw));
        double T2 = sqrt(p2*(p2-lv2)*(p2-lw2)*(p2-lvw));

        double Lvw = ((pow(lvw,2)-pow(lv1,2)-pow(lw1,2))/T1 + (pow(lvw,2)-pow(lv2,2)-pow(lw2,2))/T2)/8.0;
        Laplacian.coeffRef(LIndex,v) = Lvw;
      }
      //v=w condition
      double sum = 0;
      for(int i = 0; i < V.rows(); i++)
        sum += Laplacian.coeffRef(LIndex,i);

      Laplacian.coeffRef(LIndex,w) = -sum; //1/8 ?
      LIndex++;
    }
    return Laplacian;
}

int main(int argc, char * argv[])
{
    Eigen::MatrixXd V;
    Eigen::MatrixXi F;
    if(!igl::read_triangle_mesh(argc>1?argv[1]: "../../../../input/bunny.obj",V,F))
      std::cout<<"failed to load mesh"<<std::endl;

    eigenvectors = V.block(0,0,V.rows(), 3);
    std::vector< trimesh::triangle_t > triangles;
    int kNumVertices = V.rows();
    int kNumFaces = F.rows();
    triangles.resize( kNumFaces );
    for (int i=0; i<kNumFaces; ++i){
        triangles[i].v[0] = F(i,0);
        triangles[i].v[1] = F(i,1);
        triangles[i].v[2] = F(i,2);
    }
    std::vector< trimesh::edge_t > edges;
    trimesh::unordered_edges_from_triangles( triangles.size(), &triangles[0], edges );
    trimesh::trimesh_t mesh;
    mesh.build( kNumVertices, triangles.size(), &triangles[0], edges.size(), &edges[0] );
    
    Eigen::SparseMatrix<double> M;
    igl::massmatrix(V, F, igl::MASSMATRIX_TYPE_BARYCENTRIC, M); //unused
    std::cout<<"Calculating Laplacian...\n";
    auto Laplacian = getLaplacian(mesh,V);
    std::cout<<"Calculating Eigenvectors...\n";
    Eigen::SelfAdjointEigenSolver<Eigen::SparseMatrix<double>> eigenSolver(Laplacian.rows());
    eigenSolver.compute(Laplacian);
    eigenvectors = eigenSolver.eigenvectors();
    while(true)
    {
      igl::opengl::glfw::Viewer viewer;
      igl::opengl::glfw::imgui::ImGuiPlugin plugin;
      viewer.plugins.push_back(&plugin);
      igl::opengl::glfw::imgui::ImGuiMenu menu;
      plugin.widgets.push_back(&menu);
      viewer.data().set_mesh(V,F);
      viewer.callback_key_down = &key_down; // setting the callback
      viewer.data().show_lines = true;
      viewer.core().is_animating = true;  
      int s = 0;
      std::cout << "Select Visualisation.\n";
      std::cout << "Regular mesh: 0   Eigenvectors: 1   Heat equation: 2   Mean Curvature: 3 \n";
      std::cin >> s;
      switch(s)
      {
        case 0:
          viewer.launch();
          break;
        case 1:
          eigen(Laplacian,viewer,0);
          break;
        case 2:
          heat(Laplacian,viewer);
          break;
        case 3:
          newslug(Laplacian,viewer,V,F);
          break;
        default:
          viewer.launch();
      }
    }
}