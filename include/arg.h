/**
 * @file arg.h
 * @brief A simple command line argument parser.
 * 
 * Provides:
 *  - Global arg_parser instance `args` for defining and parsing command line arguments.
 *  - Support for basic types (int, double, string) and enum types.
 *  - All arguments are defined in the `init_args()` function, which should be called before parsing.
 */
#pragma once
#include <string>
#include <unordered_map>
#include <memory>
#include <stdexcept>
#include <sstream>
#include <vector>
#include <stdio.h>

#include "nnml.h"
#include "ops.h"


#define GB  * 1024 * 1024 * 1024


void print_logo() {
    printf("\n");
    printf("    ⚡  .-^-.        .-^-.        .-^-.    ⚡  .-^-.        .-^-.⚡\n");
    printf("       /     \\      /     \\ ⚡   /     \\      /     \\      /     \\\n");
    printf("      /  ^▽^  \\____/  o o  \\____/  o o  \\____/  o o  \\____/  > <  \\\n");
    printf("     /                ArcLight Tech Beam Activated!           ◡    \\\n");
    printf("    /_______________________________________________________________\\\n");
    printf("\n");

    printf("       █████╗ ██████╗  ██████╗ ██╗     ██╗ ██████╗ ██╗  ██╗████████╗\n");
    printf("      ██╔══██╗██╔══██╗██╔════╝ ██║     ██║██╔════╝ ██║  ██║╚══██╔══╝\n");
    printf("      ███████║██████╔╝██║      ██║     ██║██║  ███╗███████║   ██║   \n");
    printf("      ██╔══██║██╔══██╗██║      ██║     ██║██║   ██║██╔══██║   ██║   \n");
    printf("      ██║  ██║██║  ██║╚██████╗ ███████╗██║╚██████╔╝██║  ██║   ██║   \n");
    printf("      ╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚═╝ ╚═════╝ ╚═╝  ╚═╝   ╚═╝   \n");

    printf("\n");
    printf("                    ✦  ✦  ✦  ArcLight Team  ✦  ✦  ✦\n");
    printf("\n");
}


class arg_base {
public:
    std::string name;

    arg_base(const std::string & n) : name(n) {}
    virtual ~arg_base(){}

    virtual void parse(const std::string& v) = 0;
};

template<typename T>
class arg : public arg_base {
public:
    T value;
    T default_value;

    arg(std::string n, const T & def) : arg_base(n), value(def), default_value(def) {}

    void parse(const std::string & v) {
        std::stringstream ss(v);
        ss >> value;
        if(ss.fail())
            throw std::runtime_error("bad value for "+name);
    }
};

template<>
class arg<std::string> : public arg_base {
public:
    std::string value;
    std::string default_value;

    arg(std::string n, const std::string & def) : arg_base(n), value(def), default_value(def) {}

    void parse(const std::string & v)
    {
        value = v;
    }
};

template<typename enum_t>
class enum_arg : public arg_base {
public:
    enum_t value;
    enum_t default_value;

    std::unordered_map<std::string, enum_t> mapping;

    enum_arg(std::string n, enum_t def, std::unordered_map<std::string, enum_t> map)
        : arg_base(n), value(def), default_value(def), mapping(map) {}

    void parse(const std::string & v) {
        auto it = mapping.find(v);
        if(it == mapping.end())
            throw std::runtime_error("invalid enum " + v);
        value = it->second;
    }
};

class arg_parser {
public:
    std::unordered_map<std::string, arg_base*> args;
    std::vector<std::unique_ptr<arg_base> > storage;

    template<typename T>
    void add(const std::string & name,const T & def) {
        auto ptr = std::unique_ptr<arg_base>(
            new arg<T>(name, def));
        args[name] = ptr.get();
        storage.push_back(std::move(ptr));
    }

    template<typename enum_t>
    void add_enum(const std::string & name, enum_t def, std::unordered_map<std::string, enum_t> map) {
        auto ptr = std::unique_ptr<arg_base>(new enum_arg<enum_t>(name, def, map));
        args[name] = ptr.get();
        storage.push_back(std::move(ptr));
    }

    template<typename T>
    T get(const std::string & name) {
        auto base = args[name];
        auto arg_ = dynamic_cast<arg<T>*>(base);
        if(!arg_)
            throw std::runtime_error("type mismatch");
        return arg_->value;
    }

    template<typename enum_t>
    enum_t get_enum(const std::string & name) {
        auto base = args[name];
        auto arg = dynamic_cast<enum_arg<enum_t>*>(base);
        if(!arg)
            throw std::runtime_error("type mismatch");
        return arg->value;
    }

    void parse(int32_t argc, char** argv) {
        for(int32_t i = 1;i < argc; ++i) {
            std::string key = argv[i];
            if(key.rfind("--", 0) != 0)
                continue;
            key = key.substr(2);

            if(i + 1 >= argc) throw std::runtime_error("missing value");

            std::string val = argv[++i];

            auto it = args.find(key);

            if(it == args.end()) throw std::runtime_error("unknown arg " + key);

            it->second->parse(val);
        }
    }
};


extern arg_parser args;

inline void init_args() {

    args.add<bool>("asm", 1);                       // 1 is true, 0 is false
    args.add<bool>("fattn", 1);                     // 1 is true, 0 is false
    args.add<bool>("print_model", 1);               // 1 is true, 0 is false
    args.add<bool>("print_binding", 1);             // 1 is true, 0 is false
    args.add<bool>("print_kv", 1);                  // 1 is true, 0 is false
    args.add<bool>("print_perf", 1);                // 1 is true, 0 is false

    args.add<int>("threads", 48);
    args.add<int>("nodes", 4);
    args.add<int>("w_gb", 4);
    args.add<int>("a_gb", 8);
    args.add<int>("kv_gb", 2);
    args.add<int>("work_gb", 2);
    args.add<int>("max_length", 4096);
    args.add<int>("max_gen", 1024);

    args.add<std::string>("model", "/home/xyz/Qwen3-4B-Q4_0.gguf");
    args.add<std::string>("numa", "tp");          // none, tp, pp
    args.add<std::string>("prompt", "Hello!");
    
    args.add_enum<nnml_type>(
        "tk",
        NNML_TYPE_Q4_0,
        {
            {"q4_0", NNML_TYPE_Q4_0},
            {"f16",  NNML_TYPE_F16},
            {"f32",  NNML_TYPE_F32},
        }
    );
    args.add_enum<nnml_type>(
        "tv",
        NNML_TYPE_Q4_0,
        {
            {"q4_0", NNML_TYPE_Q4_0},
            {"f16",  NNML_TYPE_F16},
            {"f32",  NNML_TYPE_F32},
        }
    );

}