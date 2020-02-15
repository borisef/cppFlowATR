#include "cppflowATR/InitParams.h"

InitParams::InitParams(std::string filepath)
{
    std::ifstream file;
    file.open(filepath);

    file >> j;
    file.close();

    if (!j["info"].empty())
    {
        info = j["info"].get<dict>();
    }

    if (!j["run_params"].empty())
    {
        run_params = j["run_params"].get<dict>();
    }
    if (!j["models"].empty())
    {
        //models = new std::map<std::string, std::string>[j["models"].size()];
        models = j["models"].get<std::vector<dict>>();
        
        // for (size_t i = 0; i < j["models"].size(); i++)
        // {
        //     models.push_back(j["models"][i].get<dict>());
        // }
        int i = 0;
    }

    m_filePath = filepath;
}