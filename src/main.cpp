#ifdef _MSC_VER
#pragma warning(disable: 4244 4849 4018)
#elif defined(__GNUC__)
#pragma GCC diagnostic ignored "-Wnarrowing"
#endif // disable warnings on data type conversions

#include "scene/box.h"
#include "scene/rsm_test.h"
#include "scene/pbr_test.h"

int main(int, char**){
    // Using 4x SSAA will cost more memory and time.
    bool use4xSSAA = true;

    // Choose one scene at a time! 
    //box(use4xSSAA);
    //rsm_test(use4xSSAA);
    pbr_test(use4xSSAA);
}
