#include <iostream>
#include <fstream>
#include <iomanip>
#include <map>
#include "InitParams.h"
#include <cppflowATRInterface/Object_Detection_Types.h>

using namespace OD;

int main()
{
    InitParams a("samplejson.json");
    return 0;
}




// class InitParams
// {
// public:
//     InitParams(std::string filepath)
//     {
//         std::ifstream file;
//         file.open(filepath);

//         file >> j;
//         file.close();

//         if (!j["info"].empty())
//         {
//             info = j["info"].get<std::map<std::string, std::string>>();
//         }

//         if (!j["run_params"].empty())
//         {
//             run_params = j["run_params"].get<std::map<std::string, std::string>>();
//         }
//         if (!j["models"].empty())
//         {
//             //models = new std::map<std::string, std::string>[j["models"].size()];
//             for (size_t i = 0; i < j["models"].size(); i++)
//             {
//                 models[i] = j["models"][i].get<std::map<std::string, std::string>>();
//             }
//         }
//     }
//     std::map<std::string, std::string> info;
//     std::map<std::string, std::string> run_params;
//     std::map<std::string, std::string> models[64];

// protected:
//     json j;
// };


// struct Modell
// {
//     int id;
//     std::string task;
//     std::string filetype;
//     MB_MissionType mission;
//     std::string load_path;
//     std::string nickname;
//     int max_objects;
//     int accuracy;
//     std::string targets;
//     int height;
//     int width;
//     int max_batch_size;
//     e_OD_ColorImageType image_format;
//     int resolution;
//     std::string additional;
// };

// void from_json(const json &j, Modell &m)
// {
//     j.at("id").get_to(m.id);
//     j.at("task").get_to(m.task);
//     j.at("filetype").get_to(m.filetype);
//     j.at("mission").get_to(m.mission);
//     j.at("load_path").get_to(m.load_path);
//     j.at("nickname").get_to(m.nickname);
//     j.at("max_objects").get_to(m.max_objects);
//     j.at("accuracy").get_to(m.accuracy);
//     j.at("targets").get_to(m.targets);
//     j.at("height").get_to(m.height);
//     j.at("width").get_to(m.width);
//     j.at("max_batch_size").get_to(m.max_batch_size);
//     j.at("image_format").get_to(m.image_format);
//     j.at("resolution").get_to(m.resolution);
//     j.at("additional").get_to(m.additional);
// }

// int main()
// {
//     std::ifstream file;
//     file.open("samplejson.json");

//     json j;
//     file >> j;
//     file.close();
//     Modell models[j["models"].size()];

//     int version = j["version"];
//     std::string author = j["author"];

//    Modell models[] = j["models"];

//     for (size_t i = 0; i < j["models"].size(); i++)
//     {
//         models[i] = j["models"][i].get<Modell>();
//     }

//     std::cout << "yes it has worked thank you" << std::endl;
//     return 0;
// }
