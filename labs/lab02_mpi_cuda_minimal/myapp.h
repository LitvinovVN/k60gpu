typedef struct tag_App_Param_t {
// Input parameters:
  int ni;
  double a;
  double b;

// Output parameters:
  double sum;

// Work parameters:
  double h;
  int nsum;
} App_Param_t;

int MyAppInit(int argc, char *argv[], App_Config_t *AppCfg, App_Param_t *AppPar);

int MyAppDone(App_Config_t *AppCfg, App_Param_t *AppPar);

