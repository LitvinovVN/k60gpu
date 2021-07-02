/** CPU Laplace solver using optimized red-black Gauss–Seidel with SOR solver
 * \file main_cpu.c
 *
 * \author Kyle E. Niemeyer
 * \date 09/21/2012
 *
 * Solves Laplace's equation in 2D (e.g., heat conduction in a rectangular plate)
 * using the red-black Gauss–Seidel with sucessive overrelaxation (SOR) that has
 * been "optimized". This means that the red and black kernels only loop over
 * their respective cells, instead of over all cells and skipping even/odd cells.
 * 
 * Boundary conditions:
 * T = 0 at x = 0, x = L, y = 0
 * T = TN at y = H
 */

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <omp.h>

#include "timer.h"

/** Problem size along one side; total number of cells is this squared */
#define NUM 2048

/** Double precision */
#define DOUBLE

#ifdef DOUBLE
	#define Real double
	#define ZERO 0.0
	#define ONE 1.0
	#define TWO 2.0

 	/** SOR relaxation parameter */
	const Real omega = 1.85;
#else
	#define Real float
	#define ZERO 0.0f
	#define ONE 1.0f
	#define TWO 2.0f

	/** SOR relaxation parameter */
	const Real omega = 1.85f;
#endif

#if __STDC_VERSION__ < 199901L
	#define restrict __restrict__
#endif

#define SIZE (NUM * NUM)
#define SIZET (NUM * NUM/2 + 3*NUM + 4)

void MyRange(int np, int mp, int ia, int ib, int *i1, int *i2, int *nc)
{
  if (np<2) { *i1=ia; *i2=ib; *nc=ib-ia+1; }
  else {
    int ni, mi, nn;
    nn = ib - ia + 1; ni = nn / np;  mi = nn - ni * np;
    if (mp+1<=mi)
      { *i1 = ia + mp * (ni+1); *i2 = *i1 + ni; }
    else
      { *i1 = ia + mi * (ni+1) + (mp-mi) * ni; *i2 = *i1 + ni - 1; }
    *nc = *i2 - *i1 + 1;
  }
  return;
}

/** Function to evaluate coefficient matrix and right-hand side vector.
 * 
 * \param[in]		rowmax		number of rows
 * \param[in]		colmax		number of columns
 * \param[in]		th_cond		thermal conductivity
 * \param[in]		dx				grid size in x dimension (uniform)
 * \param[in]		dy				grid size in y dimension (uniform)
 * \param[in]		width			width of plate (z dimension)
 * \param[in]		TN				temperature at top boundary
 * \param[out]	aP				array of self coefficients
 * \param[out]	aW				array of west neighbor coefficients
 * \param[out]	aE				array of east neighbor coefficients
 * \param[out]	aS				array of south neighbor coefficients
 * \param[out]	aN				array of north neighbor coefficients
 * \param[out]	b					right-hand side array
 */

void fill_coeffs (const int nth, const int mth, const int rowmax, const int colmax,
                  Real th_cond, Real dx, Real dy, Real width, Real TN,
                  Real* aP, Real* aW, Real* aE, Real* aS, Real* aN, Real* b) {
  
  int ind, i1, i2, nc;

  MyRange(nth, mth, 0, colmax*rowmax, &i1, &i2, &nc);

  for (ind = i1; ind <= i2; ind++) {
    int col = ind / rowmax;
    int row = ind % rowmax; 
    Real SP = ZERO;
			
    b[ind] = ZERO;
			
    if (col == 0) {
      // left BC: temp = 0
      aW[ind] = ZERO;
      SP = -TWO * th_cond * width * dy / dx;
    } else {
      aW[ind] = th_cond * width * dy / dx;
    }
			
    if (col == colmax - 1) {
      // right BC: temp = 0
      aE[ind] = ZERO;
      SP = -TWO * th_cond * width * dy / dx;
    } else {
      aE[ind] = th_cond * width * dy / dx;
    }
			
    if (row == 0) {
      // bottom BC: temp = 0
      aS[ind] = ZERO;
      SP = -TWO * th_cond * width * dx / dy;
    } else {
      aS[ind] = th_cond * width * dx / dy;
    }
			
    if (row == rowmax - 1) {
      // top BC: temp = TN
      aN[ind] = ZERO;
      b[ind] = TWO * th_cond * width * dx * TN / dy;
      SP = -TWO * th_cond * width * dx / dy;
    } else {
      aN[ind] = th_cond * width * dx / dy;
    }
			
    aP[ind] = aW[ind] + aE[ind] + aS[ind] + aN[ind] - SP;
  } // end for ind

} // end fill_coeffs

///////////////////////////////////////////////////////////////////////////////

/** Function to update temperature for red cells
 * 
 * \param[in]			aP					array of self coefficients
 * \param[in]			aW					array of west neighbor coefficients
 * \param[in]			aE					array of east neighbor coefficients
 * \param[in]			aS					array of south neighbor coefficients
 * \param[in]			aN					array of north neighbor coefficients
 * \param[in]			b						right-hand side array
 * \param[in]			temp_black	temperatures of black cells, constant in this function
 * \param[inout]	temp_red		temperatures of red cells
 * \return				norm_L2			summed residuals
 */
Real red_kernel (const Real *restrict aP, const Real *restrict aW,
								 const Real *restrict aE, const Real *restrict aS, 
								 const Real *restrict aN, const Real *restrict b,
								 const Real *restrict temp_black, Real *restrict temp_red)
{
	Real norm_L2 = ZERO;
	int col, row;
	
	// loop over actual cells, skip boundary cells

	#pragma omp parallel for shared(aP, aW, aE, aS, aN, temp_black, temp_red) \
					reduction(+:norm_L2) private(col, row)

	for (col = 1; col < NUM + 1; ++col) {
		for (row = 1; row < (NUM / 2) + 1; ++row) {
			
			int ind_red = col * ((NUM / 2) + 2) + row;  		// local (red) index
			int ind = 2 * row - (col % 2) - 1 + NUM * (col - 1);	// global index
			
			Real res = b[ind] + (aW[ind] * temp_black[row + (col - 1) * ((NUM / 2) + 2)]
											   + aE[ind] * temp_black[row + (col + 1) * ((NUM / 2) + 2)]
											   + aS[ind] * temp_black[row - (col % 2) + col * ((NUM / 2) + 2)]
											   + aN[ind] * temp_black[row + ((col + 1) % 2) + col * ((NUM / 2) + 2)]);
			
			Real temp_old = temp_red[ind_red];
			temp_red[ind_red] = temp_old * (ONE - omega) + omega * (res / aP[ind]);
			
			// calculate residual
			res = temp_red[ind_red] - temp_old;
			norm_L2 += (res * res);
				
		} // end for row
	} // end for col
	
	return norm_L2;
} // end red_kernel

///////////////////////////////////////////////////////////////////////////////

/** Function to update temperature for black cells
 * 
 * \param[in]			aP					array of self coefficients
 * \param[in]			aW					array of west neighbor coefficients
 * \param[in]			aE					array of east neighbor coefficients
 * \param[in]			aS					array of south neighbor coefficients
 * \param[in]			aN					array of north neighbor coefficients
 * \param[in]			b						right-hand side array
 * \param[in]			temp_red		temperatures of red cells, constant in this function
 * \param[inout]	temp_black	temperatures of black cells
 * \return				norm_L2			variable holding summed residuals
 */
Real black_kernel (const Real *restrict aP, const Real *restrict aW,
									 const Real *restrict aE, const Real *restrict aS,
								   const Real *restrict aN, const Real *restrict b,
									 const Real *restrict temp_red, Real *restrict temp_black)
{
	Real norm_L2 = ZERO;
	int col, row;
	
	// loop over actual cells, skip boundary cells

	#pragma omp parallel for shared(aP, aW, aE, aS, aN, temp_black, temp_red) \
					reduction(+:norm_L2) private(col, row)

	for (col = 1; col < NUM + 1; ++col) {
		#pragma acc loop independent
		for (row = 1; row < (NUM / 2) + 1; ++row) {
			
			int ind_black = col * ((NUM / 2) + 2) + row;  					// local (black) index
			int ind = 2 * row - ((col + 1) % 2) - 1 + NUM * (col - 1);	// global index

			Real res = b[ind] + (aW[ind] * temp_red[row + (col - 1) * ((NUM / 2) + 2)]
											   + aE[ind] * temp_red[row + (col + 1) * ((NUM / 2) + 2)]
											   + aS[ind] * temp_red[row - ((col + 1) % 2) + col * ((NUM / 2) + 2)]
											   + aN[ind] * temp_red[row + (col % 2) + col * ((NUM / 2) + 2)]);
			
			Real temp_old = temp_black[ind_black];
			temp_black[ind_black] = temp_old * (ONE - omega) + omega * (res / aP[ind]);
			
			// calculate residual
			res = temp_black[ind_black] - temp_old;
			norm_L2 += (res * res);
			
		} // end for row
	} // end for col
	
	return norm_L2;
} // end black_kernel

///////////////////////////////////////////////////////////////////////////////

/** Main function that solves Laplace's equation in 2D (heat conduction in plate)
 * 
 * Contains iteration loop for red-black Gauss-Seidel with SOR
 */
int main (void) {
	
	// size of plate
	Real L = 1.0;
	Real H = 1.0;
	Real width = 0.01;
	
	// thermal conductivity
	Real th_cond = 1.0;
	
	// temperature at top boundary
	Real TN = 1.0;
	
	// SOR iteration tolerance
	Real tol = 1.e-6;
	
	// number of cells in x and y directions
	// including unused boundary cells
	int num_rows = (NUM / 2) + 2;
	int num_cols = NUM + 2;
	int size_temp = num_rows * num_cols;
	int size = NUM * NUM;
	
	// size of cells
	Real dx = L / NUM;
	Real dy = H / NUM;
	
	// iterations for Red-Black Gauss-Seidel with SOR
	int iter;
	int it_max = 1000000;

	// Other variables:
	int i;
	Real norm_L2 = ZERO;

	// threads number
	int nth = omp_get_max_threads();
        int mth = 0;

	// runtime:
	double runtime = 0;
	
	// allocate memory
	Real *restrict aP, *restrict aW, *restrict aE, *restrict aS, *restrict aN, *restrict b;
	Real *restrict temp_red, *restrict temp_black;
	
	// arrays of coefficients
	aP = (Real *) calloc (size, sizeof(Real));
	aW = (Real *) calloc (size, sizeof(Real));
	aE = (Real *) calloc (size, sizeof(Real));
	aS = (Real *) calloc (size, sizeof(Real));
	aN = (Real *) calloc (size, sizeof(Real));
	
	// RHS
	b = (Real *) calloc (size, sizeof(Real));
	
	// temperature arrays for red and black cells
	temp_red = (Real *) calloc (size_temp, sizeof(Real));
	temp_black = (Real *) calloc (size_temp, sizeof(Real));

	// print problem info
	fprintf(stderr,"Real size: %d\n", sizeof(Real));
	fprintf(stderr,"Problem size: %d x %d \n", NUM, NUM);
	fprintf(stderr,"Max threads: %d\n", nth);

	//////////////////////////////
	StartTimer();
	//////////////////////////////

	// set coefficients
        #pragma omp parallel
        {
            int nt = omp_get_max_threads();
            int mt = omp_get_thread_num();
            fill_coeffs (nt, mt, NUM, NUM, th_cond, dx, dy, width, TN, aP, aW, aE, aS, aN, b);
        } // end parallel
	
	/////////////////////////////////
	runtime = GetTimer();
	/////////////////////////////////

	fprintf(stderr,"Prepare time: %f s\n", runtime / 1000);

	// red-black Gauss-Seidel with SOR iteration loop

	for (iter = 1; iter <= it_max; ++iter) {
		
		norm_L2 = ZERO;
		
		// update red cells
		norm_L2 += red_kernel (aP, aW, aE, aS, aN, b, temp_black, temp_red);
		
		// update black cells
		norm_L2 += black_kernel (aP, aW, aE, aS, aN, b, temp_red, temp_black);
		
		// calculate residual
		norm_L2 = norm_L2 / ((Real)size);

		if (iter % 1000 == 0) fprintf(stderr, "%6d, %14.6le\n", iter, (double)norm_L2);
		
		// if tolerance has been reached, end SOR iterations
		if (norm_L2 < tol*tol) {
			break;
		}	
	}
	
	/////////////////////////////////
	runtime = GetTimer();
	/////////////////////////////////
	
	fprintf(stderr, "�PU-OpenMP\n");
	fprintf(stderr, "Iterations: %5d %14.6le\n", iter, (double)norm_L2);
	fprintf(stderr, "Total time: %f s\n", runtime / 1000);
	
	// write temperature data to file
	FILE * pfile;
	pfile = fopen("temp_omp.plt", "w");
	if (pfile != NULL) {
                fprintf(pfile, "TITLE = \"Temperature (K)\"\n");
                fprintf(pfile, "VARIABLES = \"X\" \"Y\" \"T\"\n");
                fprintf(pfile, "ZONE T=\"ZONE 0\" I=%d J=%d F=POINT\n", NUM, NUM);
		
		int row, col;
		for (row = 1; row < NUM + 1; ++row) {
			for (col = 1; col < NUM + 1; ++col) {
				Real x_pos = (col - 1) * dx + (dx / 2);
				Real y_pos = (row - 1) * dy + (dy / 2);
				
				if ((row + col) % 2 == 0) {
					// even, so red cell
					int ind = col * num_rows + (row + (col % 2)) / 2;
					fprintf(pfile, "%.6lf %.6lf %.6lf\n", (double)x_pos, (double)y_pos, (double)temp_red[ind]);
				} else {
					// odd, so black cell
					int ind = col * num_rows + (row + ((col + 1) % 2)) / 2;
					fprintf(pfile, "%.6lf %.6lf %.6lf\n", (double)x_pos, (double)y_pos, (double)temp_black[ind]);
				}
				
			}
			fprintf(pfile, "\n");
		}
	}
	fclose(pfile);
	
	// free memory
	free(aP);
	free(aW);
	free(aE);
	free(aS);
	free(aN);
	free(b);
	free(temp_red);
	free(temp_black);
	
	return 0;
}
