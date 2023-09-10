#include "include/App_config.hpp"
#include "include/InferenceEngine.hpp"
#include "include/Log.hpp"
#include "include/SensingEngine.hpp"

/* ************************************************************************************************
 * Global Resource
 * ************************************************************************************************
 */
void globalResourceInit_hook (void)
{
    logInit();
}

void globalResourceDestory_hook (void)
{
    logDestory();
}

/* ************************************************************************************************
 * Main
 * ************************************************************************************************
 */
int main (int argc, char** argv)
{

    globalResourceInit_hook();

    // Parallel perception sensing, synchronous in period.
    SensingEngine SE;
    SE.run();

    // Inference Engine
    InferenceEngine* IE;

#if (INFERENCE_ENGINE == RT_CPS)
    IE = new CPS_Engine(&SE);

#elif (INFERENCE_ENGINE == RT_SGE)
    IE = new SGE_Engine(&SE);

#endif
    IE->run();

    // Finish all
    log_D("main", "Finish inference engine");
    SE.stop();

    globalResourceDestory_hook();
}