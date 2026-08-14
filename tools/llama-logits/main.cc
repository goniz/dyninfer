#include <llama.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <dlfcn.h>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace fs = std::filesystem;

namespace {

constexpr const char *kTraceFormat = "dyninfer.logit-trace";
constexpr const char *kTokenFormat = "dyninfer.tokenized-prompt";
constexpr uint32_t kProtocolVersion = 1;

class Sha256 {
public:
  Sha256() {
    state_ = {0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
              0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u};
  }

  void Update(const uint8_t *data, size_t size) {
    bit_length_ += static_cast<uint64_t>(size) * 8;
    while (size > 0) {
      const size_t take = std::min(size, block_.size() - block_size_);
      std::memcpy(block_.data() + block_size_, data, take);
      block_size_ += take;
      data += take;
      size -= take;
      if (block_size_ == block_.size()) {
        Transform(block_.data());
        block_size_ = 0;
      }
    }
  }

  std::string Final() {
    block_[block_size_++] = 0x80;
    if (block_size_ > 56) {
      std::fill(block_.begin() + block_size_, block_.end(), 0);
      Transform(block_.data());
      block_size_ = 0;
    }
    std::fill(block_.begin() + block_size_, block_.begin() + 56, 0);
    for (int i = 0; i < 8; ++i) {
      block_[63 - i] = static_cast<uint8_t>(bit_length_ >> (i * 8));
    }
    Transform(block_.data());
    std::ostringstream out;
    out << std::hex << std::setfill('0');
    for (uint32_t value : state_)
      out << std::setw(8) << value;
    return out.str();
  }

private:
  static uint32_t RotateRight(uint32_t value, uint32_t count) {
    return (value >> count) | (value << (32 - count));
  }

  void Transform(const uint8_t *block) {
    static constexpr uint32_t k[64] = {
        0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu,
        0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u, 0xd807aa98u, 0x12835b01u,
        0x243185beu, 0x550c7dc3u, 0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u,
        0xc19bf174u, 0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu,
        0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau, 0x983e5152u,
        0xa831c66du, 0xb00327c8u, 0xbf597fc7u, 0xc6e00bf3u, 0xd5a79147u,
        0x06ca6351u, 0x14292967u, 0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu,
        0x53380d13u, 0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
        0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u, 0xd192e819u,
        0xd6990624u, 0xf40e3585u, 0x106aa070u, 0x19a4c116u, 0x1e376c08u,
        0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu,
        0x682e6ff3u, 0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
        0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u};
    uint32_t w[64];
    for (size_t i = 0; i < 16; ++i) {
      w[i] = (static_cast<uint32_t>(block[i * 4]) << 24) |
             (static_cast<uint32_t>(block[i * 4 + 1]) << 16) |
             (static_cast<uint32_t>(block[i * 4 + 2]) << 8) |
             static_cast<uint32_t>(block[i * 4 + 3]);
    }
    for (size_t i = 16; i < 64; ++i) {
      const uint32_t s0 = RotateRight(w[i - 15], 7) ^
                          RotateRight(w[i - 15], 18) ^ (w[i - 15] >> 3);
      const uint32_t s1 = RotateRight(w[i - 2], 17) ^
                          RotateRight(w[i - 2], 19) ^ (w[i - 2] >> 10);
      w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    uint32_t a = state_[0], b = state_[1], c = state_[2], d = state_[3];
    uint32_t e = state_[4], f = state_[5], g = state_[6], h = state_[7];
    for (size_t i = 0; i < 64; ++i) {
      const uint32_t sum1 =
          RotateRight(e, 6) ^ RotateRight(e, 11) ^ RotateRight(e, 25);
      const uint32_t choose = (e & f) ^ (~e & g);
      const uint32_t temp1 = h + sum1 + choose + k[i] + w[i];
      const uint32_t sum0 =
          RotateRight(a, 2) ^ RotateRight(a, 13) ^ RotateRight(a, 22);
      const uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
      const uint32_t temp2 = sum0 + majority;
      h = g;
      g = f;
      f = e;
      e = d + temp1;
      d = c;
      c = b;
      b = a;
      a = temp1 + temp2;
    }
    state_[0] += a;
    state_[1] += b;
    state_[2] += c;
    state_[3] += d;
    state_[4] += e;
    state_[5] += f;
    state_[6] += g;
    state_[7] += h;
  }

  std::array<uint32_t, 8> state_{};
  std::array<uint8_t, 64> block_{};
  size_t block_size_ = 0;
  uint64_t bit_length_ = 0;
};

std::string FileSha256(const fs::path &path) {
  std::ifstream input(path, std::ios::binary);
  if (!input)
    throw std::runtime_error("cannot open checkpoint for hashing: " +
                             path.string());
  Sha256 hash;
  std::array<uint8_t, 1024 * 1024> buffer{};
  while (input) {
    input.read(reinterpret_cast<char *>(buffer.data()), buffer.size());
    const std::streamsize count = input.gcount();
    if (count > 0)
      hash.Update(buffer.data(), static_cast<size_t>(count));
  }
  if (!input.eof())
    throw std::runtime_error("failed while hashing checkpoint: " +
                             path.string());
  return hash.Final();
}

std::string JsonEscape(const std::string &value) {
  std::ostringstream out;
  out << '"';
  for (unsigned char c : value) {
    switch (c) {
    case '"':
      out << "\\\"";
      break;
    case '\\':
      out << "\\\\";
      break;
    case '\b':
      out << "\\b";
      break;
    case '\f':
      out << "\\f";
      break;
    case '\n':
      out << "\\n";
      break;
    case '\r':
      out << "\\r";
      break;
    case '\t':
      out << "\\t";
      break;
    default:
      if (c < 0x20) {
        out << "\\u" << std::hex << std::setw(4) << std::setfill('0')
            << static_cast<int>(c) << std::dec;
      } else {
        out << static_cast<char>(c);
      }
    }
  }
  out << '"';
  return out.str();
}

template <typename T> std::string JsonArray(const std::vector<T> &values) {
  std::ostringstream out;
  out << '[';
  for (size_t i = 0; i < values.size(); ++i) {
    if (i)
      out << ',';
    out << values[i];
  }
  out << ']';
  return out.str();
}

struct ParsedArgs {
  std::unordered_map<std::string, std::string> values;
  std::unordered_set<std::string> flags;

  const std::string &Require(const std::string &name) const {
    auto iterator = values.find(name);
    if (iterator == values.end())
      throw std::runtime_error("missing required option " + name);
    return iterator->second;
  }

  std::string Get(const std::string &name, std::string fallback) const {
    auto iterator = values.find(name);
    return iterator == values.end() ? std::move(fallback) : iterator->second;
  }

  bool Has(const std::string &name) const { return flags.count(name) != 0; }
};

ParsedArgs ParseArgs(int argc, char **argv, int start) {
  ParsedArgs parsed;
  for (int i = start; i < argc; ++i) {
    std::string option = argv[i];
    if (option.empty() || option.rfind("--", 0) != 0) {
      throw std::runtime_error("unexpected positional argument: " + option);
    }
    if (option == "--parse-special") {
      parsed.flags.insert(option);
      continue;
    }
    if (i + 1 >= argc)
      throw std::runtime_error("missing value for " + option);
    if (!parsed.values.emplace(option, argv[++i]).second) {
      throw std::runtime_error("duplicate option " + option);
    }
  }
  return parsed;
}

uint32_t ParseU32(const std::string &value, const std::string &option) {
  if (value.empty() || value[0] == '-')
    throw std::runtime_error(option + " is not a u32: " + value);
  size_t consumed = 0;
  const unsigned long parsed = std::stoul(value, &consumed, 10);
  if (consumed != value.size() ||
      parsed > std::numeric_limits<uint32_t>::max()) {
    throw std::runtime_error(option + " is not a u32: " + value);
  }
  return static_cast<uint32_t>(parsed);
}

int32_t ParseI32(const std::string &value, const std::string &option) {
  size_t consumed = 0;
  const long parsed = std::stol(value, &consumed, 10);
  if (consumed != value.size() ||
      parsed < std::numeric_limits<int32_t>::min() ||
      parsed > std::numeric_limits<int32_t>::max()) {
    throw std::runtime_error(option + " is not an i32: " + value);
  }
  return static_cast<int32_t>(parsed);
}

std::vector<llama_token> ParseTokens(const std::string &value,
                                     const std::string &option) {
  if (value.empty())
    throw std::runtime_error(option + " is empty");
  std::vector<llama_token> tokens;
  size_t start = 0;
  while (start <= value.size()) {
    const size_t comma = value.find(',', start);
    const std::string item =
        value.substr(start, comma == std::string::npos ? comma : comma - start);
    const uint32_t token = ParseU32(item, option);
    if (token >
        static_cast<uint32_t>(std::numeric_limits<llama_token>::max())) {
      throw std::runtime_error(option + " token exceeds llama_token range");
    }
    tokens.push_back(static_cast<llama_token>(token));
    if (comma == std::string::npos)
      break;
    start = comma + 1;
  }
  return tokens;
}

struct BackendGuard {
  BackendGuard() { llama_backend_init(); }
  ~BackendGuard() { llama_backend_free(); }
};

struct ModelDeleter {
  void operator()(llama_model *model) const { llama_model_free(model); }
};
struct ContextDeleter {
  void operator()(llama_context *context) const { llama_free(context); }
};
using ModelPtr = std::unique_ptr<llama_model, ModelDeleter>;
using ContextPtr = std::unique_ptr<llama_context, ContextDeleter>;

struct BatchGuard {
  explicit BatchGuard(int32_t capacity)
      : batch(llama_batch_init(capacity, 0, 1)) {
    if (!batch.token || !batch.pos || !batch.n_seq_id || !batch.seq_id ||
        !batch.logits) {
      llama_batch_free(batch);
      throw std::runtime_error("llama_batch_init failed");
    }
  }
  ~BatchGuard() { llama_batch_free(batch); }
  llama_batch batch;
};

std::string LoadedLibraryPath() {
  Dl_info info{};
  if (dladdr(reinterpret_cast<void *>(llama_model_load_from_file), &info) !=
          0 &&
      info.dli_fname) {
    std::error_code error;
    const fs::path canonical = fs::canonical(info.dli_fname, error);
    return error ? std::string(info.dli_fname) : canonical.string();
  }
  return "unavailable";
}

std::string LlamaBuildNumber() {
  const std::string filename =
      fs::path(LoadedLibraryPath()).filename().string();
  const size_t separator = filename.rfind('.');
  if (separator == std::string::npos || separator + 1 == filename.size())
    return "unavailable";
  const std::string suffix = filename.substr(separator + 1);
  if (!std::all_of(suffix.begin(), suffix.end(),
                   [](unsigned char value) { return std::isdigit(value); }))
    return "unavailable";
  return suffix;
}

std::string ModelDescription(const llama_model *model) {
  std::array<char, 1024> buffer{};
  const int32_t count = llama_model_desc(model, buffer.data(), buffer.size());
  if (count < 0)
    return "unavailable";
  return std::string(buffer.data(), std::min<size_t>(static_cast<size_t>(count),
                                                     buffer.size() - 1));
}

std::vector<llama_token> Tokenize(const llama_vocab *vocab,
                                  const std::string &text, bool add_special,
                                  bool parse_special) {
  if (text.size() > static_cast<size_t>(std::numeric_limits<int32_t>::max()))
    throw std::runtime_error(
        "prompt text exceeds the llama_tokenize length range");
  int32_t count =
      llama_tokenize(vocab, text.data(), static_cast<int32_t>(text.size()),
                     nullptr, 0, add_special, parse_special);
  if (count == std::numeric_limits<int32_t>::min()) {
    throw std::runtime_error("tokenization size overflow");
  }
  const int32_t capacity = count < 0 ? -count : count;
  std::vector<llama_token> tokens(static_cast<size_t>(capacity));
  count = llama_tokenize(vocab, text.data(), static_cast<int32_t>(text.size()),
                         tokens.data(), capacity, add_special, parse_special);
  if (count < 0)
    throw std::runtime_error("llama_tokenize failed after sizing");
  tokens.resize(static_cast<size_t>(count));
  if (tokens.empty())
    throw std::runtime_error("tokenization produced no tokens");
  return tokens;
}

void WriteTextAtomically(const fs::path &destination,
                         const std::string &contents) {
  if (!destination.parent_path().empty())
    fs::create_directories(destination.parent_path());
  const fs::path temporary =
      destination.string() + ".tmp-" + std::to_string(::getpid());
  if (fs::exists(temporary))
    throw std::runtime_error("temporary output already exists: " +
                             temporary.string());
  try {
    std::ofstream output(temporary,
                         std::ios::binary | std::ios::out | std::ios::trunc);
    if (!output)
      throw std::runtime_error("cannot create output: " + temporary.string());
    output << contents;
    output.flush();
    if (!output)
      throw std::runtime_error("cannot write output: " + temporary.string());
    output.close();
    fs::rename(temporary, destination);
  } catch (...) {
    std::error_code ignored;
    fs::remove(temporary, ignored);
    throw;
  }
}

int TokenizeMode(const ParsedArgs &args) {
  const fs::path checkpoint = fs::canonical(args.Require("--checkpoint"));
  const fs::path output = args.Require("--output");
  const std::string prompt = args.Require("--prompt");
  const std::string add_special_option = args.Get("--add-special", "auto");
  if (add_special_option != "auto" && add_special_option != "yes" &&
      add_special_option != "no") {
    throw std::runtime_error("--add-special must be auto, yes, or no");
  }
  const bool add_special = add_special_option != "no";
  BackendGuard backend;
  llama_model_params params = llama_model_default_params();
  params.vocab_only = true;
  params.n_gpu_layers = 0;
  ModelPtr model(llama_model_load_from_file(checkpoint.c_str(), params));
  if (!model)
    throw std::runtime_error(
        "llama_model_load_from_file failed in vocab-only mode");
  const llama_vocab *vocab = llama_model_get_vocab(model.get());
  if (!vocab)
    throw std::runtime_error("llama_model_get_vocab returned null");
  const auto tokens =
      Tokenize(vocab, prompt, add_special, args.Has("--parse-special"));
  const int32_t vocabulary_size = llama_vocab_n_tokens(vocab);
  for (llama_token token : tokens) {
    if (token < 0 || token >= vocabulary_size)
      throw std::runtime_error("tokenizer returned invalid token ID");
  }
  std::ostringstream json;
  json << "{\n"
       << "  \"format\": " << JsonEscape(kTokenFormat) << ",\n"
       << "  \"version\": " << kProtocolVersion << ",\n"
       << "  \"checkpoint_sha256\": " << JsonEscape(FileSha256(checkpoint))
       << ",\n"
       << "  \"prompt_tokens\": " << JsonArray(tokens) << ",\n"
       << "  \"engine_metadata\": {\n"
       << "    \"engine\": \"llama.cpp\",\n"
       << "    \"loaded_libllama_path\": " << JsonEscape(LoadedLibraryPath())
       << ",\n"
       << "    \"build_number\": " << JsonEscape(LlamaBuildNumber()) << ",\n"
       << "    \"system_info\": " << JsonEscape(llama_print_system_info())
       << ",\n"
       << "    \"model_description\": "
       << JsonEscape(ModelDescription(model.get())) << ",\n"
       << "    \"add_special\": " << JsonEscape(add_special_option) << ",\n"
       << "    \"parse_special\": "
       << (args.Has("--parse-special") ? "true" : "false") << "\n"
       << "  }\n"
       << "}\n";
  WriteTextAtomically(output, json.str());
  return 0;
}

ggml_type ParseKvType(const std::string &type) {
  if (type == "f32")
    return GGML_TYPE_F32;
  if (type == "f16")
    return GGML_TYPE_F16;
  if (type == "bf16")
    return GGML_TYPE_BF16;
  throw std::runtime_error("--kv-type must be f32, f16, or bf16");
}

llama_flash_attn_type ParseFlashAttention(const std::string &mode) {
  if (mode == "off")
    return LLAMA_FLASH_ATTN_TYPE_DISABLED;
  if (mode == "on")
    return LLAMA_FLASH_ATTN_TYPE_ENABLED;
  if (mode == "auto")
    return LLAMA_FLASH_ATTN_TYPE_AUTO;
  throw std::runtime_error("--flash-attn must be off, on, or auto");
}

std::string Lower(std::string value) {
  std::transform(
      value.begin(), value.end(), value.begin(),
      [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return value;
}

ggml_backend_dev_t SelectDevice(const std::string &requested) {
  const std::string needle = Lower(requested);
  for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
    ggml_backend_dev_t device = ggml_backend_dev_get(i);
    const std::string name =
        ggml_backend_dev_name(device) ? ggml_backend_dev_name(device) : "";
    const std::string description = ggml_backend_dev_description(device)
                                        ? ggml_backend_dev_description(device)
                                        : "";
    if (Lower(name).find(needle) != std::string::npos ||
        Lower(description).find(needle) != std::string::npos) {
      return device;
    }
  }
  throw std::runtime_error("no llama.cpp backend device matches: " + requested);
}

struct OutputRow {
  std::string phase;
  uint64_t position;
  llama_token input_token;
  llama_token argmax;
};

llama_token WriteLogitRow(llama_context *context, int32_t vocabulary_size,
                          std::ofstream &output) {
  float *logits = llama_get_logits_ith(context, -1);
  if (!logits)
    throw std::runtime_error("llama_get_logits_ith returned null");
  llama_token best = 0;
  float best_value = -std::numeric_limits<float>::infinity();
  for (int32_t token = 0; token < vocabulary_size; ++token) {
    const float value = logits[token];
    if (!std::isfinite(value)) {
      throw std::runtime_error(
          "llama.cpp returned a non-finite logit for token " +
          std::to_string(token));
    }
    if (value > best_value) {
      best_value = value;
      best = token;
    }
    uint32_t bits;
    static_assert(sizeof(bits) == sizeof(value), "F32 size mismatch");
    std::memcpy(&bits, &value, sizeof(bits));
    const char bytes[4] = {static_cast<char>(bits & 0xff),
                           static_cast<char>((bits >> 8) & 0xff),
                           static_cast<char>((bits >> 16) & 0xff),
                           static_cast<char>((bits >> 24) & 0xff)};
    output.write(bytes, sizeof(bytes));
  }
  if (!output)
    throw std::runtime_error("failed to write logits.f32le");
  return best;
}

void FillPromptBatch(llama_batch &batch,
                     const std::vector<llama_token> &tokens) {
  batch.n_tokens = static_cast<int32_t>(tokens.size());
  for (int32_t i = 0; i < batch.n_tokens; ++i) {
    batch.token[i] = tokens[static_cast<size_t>(i)];
    batch.pos[i] = i;
    batch.n_seq_id[i] = 1;
    batch.seq_id[i][0] = 0;
    batch.logits[i] = i + 1 == batch.n_tokens ? 1 : 0;
  }
}

void FillDecodeBatch(llama_batch &batch, llama_token token,
                     llama_pos position) {
  batch.n_tokens = 1;
  batch.token[0] = token;
  batch.pos[0] = position;
  batch.n_seq_id[0] = 1;
  batch.seq_id[0][0] = 0;
  batch.logits[0] = 1;
}

fs::path TemporaryTraceDirectory(const fs::path &destination) {
  if (fs::exists(destination))
    throw std::runtime_error("trace output already exists: " +
                             destination.string());
  const fs::path parent = destination.parent_path().empty()
                              ? fs::path(".")
                              : destination.parent_path();
  fs::create_directories(parent);
  for (uint32_t counter = 0; counter < 1000; ++counter) {
    fs::path candidate =
        parent / ("." + destination.filename().string() + ".tmp-" +
                  std::to_string(::getpid()) + "-" + std::to_string(counter));
    std::error_code error;
    if (fs::create_directory(candidate, error))
      return candidate;
    if (error && error != std::errc::file_exists) {
      throw std::runtime_error("cannot create temporary trace directory: " +
                               error.message());
    }
  }
  throw std::runtime_error("could not allocate a temporary trace directory");
}

int RunMode(const ParsedArgs &args) {
  const fs::path checkpoint = fs::canonical(args.Require("--checkpoint"));
  const fs::path destination = args.Require("--output-dir");
  const std::vector<llama_token> prompt =
      ParseTokens(args.Require("--prompt-tokens"), "--prompt-tokens");
  const uint32_t requested_context =
      ParseU32(args.Require("--n-ctx"), "--n-ctx");
  if (requested_context >
      static_cast<uint32_t>(std::numeric_limits<int32_t>::max()))
    throw std::runtime_error("--n-ctx exceeds the llama_pos range");
  if (requested_context < prompt.size())
    throw std::runtime_error("--n-ctx is shorter than the prompt");
  const std::string kv_name = args.Require("--kv-type");
  const ggml_type kv_type = ParseKvType(kv_name);
  const std::string device_name = args.Get("--device", "cpu");
  const bool cpu = Lower(device_name) == "cpu";
  const int32_t gpu_layers =
      ParseI32(args.Get("--gpu-layers", "0"), "--gpu-layers");
  if (cpu && gpu_layers != 0)
    throw std::runtime_error("CPU mode requires --gpu-layers 0");
  const std::string flash_name = args.Get("--flash-attn", "off");
  const llama_flash_attn_type flash = ParseFlashAttention(flash_name);
  const int32_t threads = args.values.count("--threads")
                              ? ParseI32(args.Require("--threads"), "--threads")
                              : 0;
  if (args.values.count("--threads") && threads <= 0)
    throw std::runtime_error("--threads must be positive");
  const bool explicit_decode = args.values.count("--decode-tokens") != 0;
  if (explicit_decode == (args.values.count("--decode-steps") != 0)) {
    throw std::runtime_error(
        "provide exactly one of --decode-tokens or --decode-steps");
  }
  std::vector<llama_token> decode_inputs;
  uint32_t decode_steps = 0;
  if (explicit_decode) {
    decode_inputs =
        ParseTokens(args.Require("--decode-tokens"), "--decode-tokens");
    decode_steps = static_cast<uint32_t>(decode_inputs.size());
  } else {
    decode_steps = ParseU32(args.Require("--decode-steps"), "--decode-steps");
  }
  if (static_cast<uint64_t>(prompt.size()) + decode_steps > requested_context) {
    throw std::runtime_error("--n-ctx cannot fit prompt + decode steps");
  }

  BackendGuard backend;
  std::vector<ggml_backend_dev_t> devices;
  llama_model_params model_params = llama_model_default_params();
  model_params.n_gpu_layers = gpu_layers;
  ggml_backend_dev_t selected =
      cpu ? ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU)
          : SelectDevice(device_name);
  if (!selected)
    throw std::runtime_error("llama.cpp CPU backend device is unavailable");
  devices = {selected, nullptr};
  model_params.devices = devices.data();
  ModelPtr model(llama_model_load_from_file(checkpoint.c_str(), model_params));
  if (!model)
    throw std::runtime_error("llama_model_load_from_file failed");
  const llama_vocab *vocab = llama_model_get_vocab(model.get());
  if (!vocab)
    throw std::runtime_error("llama_model_get_vocab returned null");
  const int32_t vocabulary_size = llama_vocab_n_tokens(vocab);
  if (vocabulary_size <= 0)
    throw std::runtime_error("model vocabulary is empty");
  auto validate_tokens = [&](const std::vector<llama_token> &tokens,
                             const char *kind) {
    for (size_t i = 0; i < tokens.size(); ++i) {
      if (tokens[i] < 0 || tokens[i] >= vocabulary_size) {
        throw std::runtime_error(std::string(kind) + " token " +
                                 std::to_string(i) +
                                 " is outside the model vocabulary");
      }
    }
  };
  validate_tokens(prompt, "prompt");
  if (explicit_decode)
    validate_tokens(decode_inputs, "decode");

  llama_context_params context_params = llama_context_default_params();
  context_params.n_ctx = requested_context;
  context_params.n_batch = static_cast<uint32_t>(prompt.size());
  context_params.n_ubatch = static_cast<uint32_t>(prompt.size());
  context_params.n_seq_max = 1;
  context_params.n_outputs_max = 1;
  context_params.type_k = kv_type;
  context_params.type_v = kv_type;
  context_params.flash_attn_type = flash;
  context_params.offload_kqv = !cpu;
  context_params.op_offload = !cpu;
  if (threads > 0) {
    context_params.n_threads = threads;
    context_params.n_threads_batch = threads;
  }
  ContextPtr context(llama_init_from_model(model.get(), context_params));
  if (!context)
    throw std::runtime_error(
        "llama_init_from_model failed (context/KV allocation)");

  const fs::path temporary = TemporaryTraceDirectory(destination);
  try {
    std::ofstream logits(temporary / "logits.f32le",
                         std::ios::binary | std::ios::out | std::ios::trunc);
    if (!logits)
      throw std::runtime_error("cannot create logits.f32le");
    BatchGuard batch(static_cast<int32_t>(prompt.size()));
    FillPromptBatch(batch.batch, prompt);
    const int32_t prefill_status = llama_decode(context.get(), batch.batch);
    if (prefill_status != 0) {
      throw std::runtime_error("llama_decode(prompt) failed with status " +
                               std::to_string(prefill_status));
    }
    std::vector<OutputRow> rows;
    llama_token current_argmax =
        WriteLogitRow(context.get(), vocabulary_size, logits);
    rows.push_back(
        {"prefill", prompt.size() - 1, prompt.back(), current_argmax});
    for (uint32_t step = 0; step < decode_steps; ++step) {
      const llama_token input =
          explicit_decode ? decode_inputs[step] : current_argmax;
      if (!explicit_decode)
        decode_inputs.push_back(input);
      FillDecodeBatch(batch.batch, input,
                      static_cast<llama_pos>(prompt.size() + step));
      const int32_t decode_status = llama_decode(context.get(), batch.batch);
      if (decode_status != 0) {
        throw std::runtime_error("llama_decode(token) failed at step " +
                                 std::to_string(step) + " with status " +
                                 std::to_string(decode_status));
      }
      current_argmax = WriteLogitRow(context.get(), vocabulary_size, logits);
      rows.push_back({"decode", prompt.size() + step, input, current_argmax});
    }
    logits.flush();
    if (!logits)
      throw std::runtime_error("failed to flush logits.f32le");
    logits.close();

    const std::string actual_device =
        std::string(ggml_backend_dev_name(devices[0])) + " (" +
        ggml_backend_dev_description(devices[0]) + ")";
    std::ostringstream json;
    json << "{\n"
         << "  \"format\": " << JsonEscape(kTraceFormat) << ",\n"
         << "  \"version\": " << kProtocolVersion << ",\n"
         << "  \"engine\": \"llama.cpp\",\n"
         << "  \"checkpoint_sha256\": " << JsonEscape(FileSha256(checkpoint))
         << ",\n"
         << "  \"vocab_size\": " << vocabulary_size << ",\n"
         << "  \"prompt_tokens\": " << JsonArray(prompt) << ",\n"
         << "  \"decode_inputs\": " << JsonArray(decode_inputs) << ",\n"
         << "  \"logits_dtype\": \"f32\",\n"
         << "  \"logits_byte_order\": \"little\",\n"
         << "  \"rows\": [\n";
    for (size_t i = 0; i < rows.size(); ++i) {
      const OutputRow &row = rows[i];
      json << "    {\"phase\": " << JsonEscape(row.phase)
           << ", \"position\": " << row.position
           << ", \"input_token\": " << row.input_token
           << ", \"argmax\": " << row.argmax;
      if (i < decode_inputs.size())
        json << ", \"forced_token\": " << decode_inputs[i];
      json << "}" << (i + 1 == rows.size() ? "\n" : ",\n");
    }
    json << "  ],\n"
         << "  \"logits_file\": \"logits.f32le\",\n"
         << "  \"engine_metadata\": {\n"
         << "    \"loaded_libllama_path\": " << JsonEscape(LoadedLibraryPath())
         << ",\n"
         << "    \"build_number\": " << JsonEscape(LlamaBuildNumber()) << ",\n"
         << "    \"system_info\": " << JsonEscape(llama_print_system_info())
         << ",\n"
         << "    \"model_description\": "
         << JsonEscape(ModelDescription(model.get())) << ",\n"
         << "    \"model_size_bytes\": " << llama_model_size(model.get())
         << ",\n"
         << "    \"model_parameter_count\": "
         << llama_model_n_params(model.get()) << ",\n"
         << "    \"device\": " << JsonEscape(actual_device) << ",\n"
         << "    \"gpu_layers\": " << gpu_layers << ",\n"
         << "    \"threads\": " << llama_n_threads(context.get()) << ",\n"
         << "    \"threads_batch\": " << llama_n_threads_batch(context.get())
         << ",\n"
         << "    \"n_ctx\": " << llama_n_ctx(context.get()) << ",\n"
         << "    \"n_batch\": " << llama_n_batch(context.get()) << ",\n"
         << "    \"n_ubatch\": " << llama_n_ubatch(context.get()) << ",\n"
         << "    \"flash_attention\": " << JsonEscape(flash_name) << ",\n"
         << "    \"kv_type_k\": " << JsonEscape(kv_name) << ",\n"
         << "    \"kv_type_v\": " << JsonEscape(kv_name) << "\n"
         << "  }\n"
         << "}\n";
    std::ofstream header(temporary / "trace.json",
                         std::ios::binary | std::ios::out | std::ios::trunc);
    if (!header)
      throw std::runtime_error("cannot create trace.json");
    header << json.str();
    header.flush();
    if (!header)
      throw std::runtime_error("failed to write trace.json");
    header.close();
    fs::rename(temporary, destination);
  } catch (...) {
    std::error_code ignored;
    fs::remove_all(temporary, ignored);
    throw;
  }
  return 0;
}

void Usage() {
  std::cerr
      << "usage:\n"
      << "  dyninfer-llama-logits tokenize --checkpoint GGUF --prompt TEXT "
         "--add-special auto|yes|no --output FILE [--parse-special]\n"
      << "  dyninfer-llama-logits run --checkpoint GGUF --output-dir DIR "
         "--prompt-tokens IDS --n-ctx N --kv-type TYPE --device DEVICE "
         "--gpu-layers N --flash-attn MODE (--decode-steps N | --decode-tokens "
         "IDS) [--threads N]\n";
}

} // namespace

int main(int argc, char **argv) {
  try {
    if (argc < 2) {
      Usage();
      return 2;
    }
    const std::string mode = argv[1];
    const ParsedArgs args = ParseArgs(argc, argv, 2);
    if (mode == "tokenize")
      return TokenizeMode(args);
    if (mode == "run")
      return RunMode(args);
    Usage();
    throw std::runtime_error("unknown mode: " + mode);
  } catch (const std::exception &error) {
    std::cerr << "dyninfer-llama-logits: " << error.what() << '\n';
    return 1;
  }
}
