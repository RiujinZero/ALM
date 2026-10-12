// Native binary checkpoint format for GrowingVisionOrganism.
//
// Same-machine, same-build use only (raw struct layout, host endianness) --
// mirrors alm/serialize.hpp's own contract for the byte cortex. Unlike the
// byte cortex, every table here is a dense, fixed-stride array (no hash
// maps, no variable-length rows), so this format only needs to write each
// column's *live* rows -- rows beyond `live()` are unused capacity, not
// state, and are not serialized.
//
// Columns are not assumed identical: they only match right after a sleep()
// call (see GrowingVisionOrganism::memory_bytes's own comment on this), and
// a checkpoint must be safe to take mid-wake, before the columns have
// re-converged. Every column is written and read independently.
#pragma once

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

#include "alm/vision.hpp"

namespace alm {

struct VisionIO {
    static constexpr uint32_t kMagic = 0x414C5631;  // "ALV1"
    static constexpr uint32_t kVersion = 2;  // v2 adds episodic recall state

    static std::vector<uint8_t> save(const GrowingVisionOrganism& organism) {
        std::vector<uint8_t> out;
        out.reserve(1 << 20);
        write_u32(out, kMagic);
        write_u32(out, kVersion);
        write_config(out, organism.config_);

        write_i32(out, organism.dimensions_);
        write_i64(out, organism.steps_);
        write_i64(out, organism.sleep_cycles_);
        write_i64(out, organism.external_updates_);
        write_i64(out, organism.visual_births_);
        write_i64(out, organism.releases_);
        write_i32(out, organism.base_rows_);

        write_float_vector(out, organism.slow_context_);
        write_float_vector(out, organism.label_counts_);

        write_i32(out, static_cast<int32_t>(organism.columns_.size()));
        for (const VisionColumn& column : organism.columns_) write_column(out, column);

        write_i64(out, organism.episodic_step_);
        if (organism.config_.episodic_enabled) {
            write_i32(out, static_cast<int32_t>(organism.episodic_label_.size()));
            write_raw(out, organism.episodic_label_.data(), organism.episodic_label_.size());
            write_raw(out, organism.episodic_last_touch_.data(), organism.episodic_last_touch_.size());
            write_raw(out, organism.episodic_refresh_.data(), organism.episodic_refresh_.size());
            write_raw(out, organism.episodic_used_.data(), organism.episodic_used_.size());
        }
        return out;
    }

    static GrowingVisionOrganism load(const uint8_t* data, size_t size) {
        Reader reader{data, size, 0};
        if (read_u32(reader) != kMagic) throw std::runtime_error("bad ALM vision checkpoint magic");
        if (read_u32(reader) != kVersion)
            throw std::runtime_error("unsupported ALM vision checkpoint version");
        VisionConfig config = read_config(reader);
        GrowingVisionOrganism organism(config);

        const int32_t dimensions = read_i32(reader);
        if (dimensions != organism.dimensions_) {
            throw std::runtime_error("ALM vision checkpoint dimensions do not match config");
        }
        organism.steps_ = read_i64(reader);
        organism.sleep_cycles_ = read_i64(reader);
        organism.external_updates_ = read_i64(reader);
        organism.visual_births_ = read_i64(reader);
        organism.releases_ = read_i64(reader);
        organism.base_rows_ = read_i32(reader);

        read_float_vector(reader, organism.slow_context_);
        read_float_vector(reader, organism.label_counts_);

        const int32_t column_count = read_i32(reader);
        if (column_count != static_cast<int32_t>(organism.columns_.size())) {
            throw std::runtime_error("ALM vision checkpoint column count does not match config");
        }
        for (VisionColumn& column : organism.columns_) read_column(reader, column);

        organism.episodic_step_ = read_i64(reader);
        if (organism.config_.episodic_enabled) {
            const int32_t label_count = read_i32(reader);
            if (label_count != static_cast<int32_t>(organism.episodic_label_.size())) {
                throw std::runtime_error(
                    "ALM vision checkpoint episodic bucket table size does not match config");
            }
            read_raw(reader, organism.episodic_label_.data(), organism.episodic_label_.size());
            read_raw(reader, organism.episodic_last_touch_.data(),
                     organism.episodic_last_touch_.size());
            read_raw(reader, organism.episodic_refresh_.data(), organism.episodic_refresh_.size());
            read_raw(reader, organism.episodic_used_.data(), organism.episodic_used_.size());
        }
        return organism;
    }

private:
    struct Reader {
        const uint8_t* data;
        size_t size;
        size_t offset;
    };

    static void need(const Reader& reader, size_t bytes) {
        if (reader.offset + bytes > reader.size)
            throw std::runtime_error("truncated ALM vision checkpoint");
    }

    template <typename T>
    static void write_pod(std::vector<uint8_t>& out, T value) {
        const size_t at = out.size();
        out.resize(at + sizeof(T));
        std::memcpy(out.data() + at, &value, sizeof(T));
    }
    template <typename T>
    static T read_pod(Reader& reader) {
        need(reader, sizeof(T));
        T value;
        std::memcpy(&value, reader.data + reader.offset, sizeof(T));
        reader.offset += sizeof(T);
        return value;
    }

    static void write_u32(std::vector<uint8_t>& out, uint32_t v) { write_pod(out, v); }
    static void write_i32(std::vector<uint8_t>& out, int32_t v) { write_pod(out, v); }
    static void write_i64(std::vector<uint8_t>& out, int64_t v) { write_pod(out, v); }
    static void write_f64(std::vector<uint8_t>& out, double v) { write_pod(out, v); }
    static uint32_t read_u32(Reader& r) { return read_pod<uint32_t>(r); }
    static int32_t read_i32(Reader& r) { return read_pod<int32_t>(r); }
    static int64_t read_i64(Reader& r) { return read_pod<int64_t>(r); }
    static double read_f64(Reader& r) { return read_pod<double>(r); }

    static void write_config(std::vector<uint8_t>& out, const VisionConfig& c) {
        write_i32(out, c.classes);
        write_i32(out, c.image_size);
        write_i32(out, c.columns);
        write_f64(out, c.context_decay);
        write_i32(out, c.sleep_interval);
        write_i64(out, c.max_body_bytes);
        write_f64(out, c.learning_rate);
        write_i32(out, c.maximum_synapses);
        write_i32(out, c.maximum_visual_assemblies);
        write_f64(out, c.visual_birth_activation);
        write_i32(out, c.visual_routing_width);
        write_i32(out, c.relation_sketch_words);
        write_f64(out, c.relation_turnover_fraction);
        write_i32(out, c.episodic_enabled ? 1 : 0);
        write_i32(out, c.episodic_lsh_bits);
        write_i32(out, c.episodic_probe_radius);
        write_i32(out, c.episodic_chain_slots);
        write_i32(out, c.episodic_tier3_refresh_threshold);
        write_f64(out, c.episodic_strength);
    }
    static VisionConfig read_config(Reader& r) {
        VisionConfig c;
        c.classes = read_i32(r);
        c.image_size = read_i32(r);
        c.columns = read_i32(r);
        c.context_decay = read_f64(r);
        c.sleep_interval = read_i32(r);
        c.max_body_bytes = read_i64(r);
        c.learning_rate = read_f64(r);
        c.maximum_synapses = read_i32(r);
        c.maximum_visual_assemblies = read_i32(r);
        c.visual_birth_activation = read_f64(r);
        c.visual_routing_width = read_i32(r);
        c.relation_sketch_words = read_i32(r);
        c.relation_turnover_fraction = read_f64(r);
        c.episodic_enabled = read_i32(r) != 0;
        c.episodic_lsh_bits = read_i32(r);
        c.episodic_probe_radius = read_i32(r);
        c.episodic_chain_slots = read_i32(r);
        c.episodic_tier3_refresh_threshold = read_i32(r);
        c.episodic_strength = read_f64(r);
        return c;
    }

    static void write_float_vector(std::vector<uint8_t>& out, const std::vector<float>& values) {
        write_i64(out, static_cast<int64_t>(values.size()));
        if (!values.empty()) {
            const size_t at = out.size();
            out.resize(at + values.size() * sizeof(float));
            std::memcpy(out.data() + at, values.data(), values.size() * sizeof(float));
        }
    }
    static void read_float_vector(Reader& reader, std::vector<float>& values) {
        const int64_t count = read_i64(reader);
        need(reader, static_cast<size_t>(count) * sizeof(float));
        values.resize(static_cast<size_t>(count));
        if (count) {
            std::memcpy(values.data(), reader.data + reader.offset,
                        static_cast<size_t>(count) * sizeof(float));
            reader.offset += static_cast<size_t>(count) * sizeof(float);
        }
    }

    // Only the live() rows of each dense array -- rows beyond that are
    // unused pre-allocated capacity, not state.
    static void write_column(std::vector<uint8_t>& out, const VisionColumn& column) {
        const int32_t live = column.live();
        write_i32(out, live);
        write_i64(out, column.births());
        const int32_t synapses = column.synapses();
        const int32_t classes = column.classes();
        const int32_t words = column.relation_words();
        write_raw(out, column.indices_data(), static_cast<size_t>(live) * synapses);
        write_raw(out, column.weights_data(), static_cast<size_t>(live) * synapses);
        write_raw(out, column.labels_data(), static_cast<size_t>(live) * classes);
        write_raw(out, column.observations_data(), static_cast<size_t>(live));
        write_raw(out, column.relations_data(), static_cast<size_t>(live) * words);
    }
    static void read_column(Reader& reader, VisionColumn& column) {
        const int32_t live = read_i32(reader);
        const int64_t births = read_i64(reader);
        const int32_t synapses = column.synapses();
        const int32_t classes = column.classes();
        const int32_t words = column.relation_words();
        std::vector<int32_t> indices(static_cast<size_t>(live) * synapses);
        std::vector<float> weights(static_cast<size_t>(live) * synapses);
        std::vector<float> labels(static_cast<size_t>(live) * classes);
        std::vector<int32_t> observations(static_cast<size_t>(live));
        std::vector<uint64_t> relations(static_cast<size_t>(live) * words);
        read_raw(reader, indices.data(), indices.size());
        read_raw(reader, weights.data(), weights.size());
        read_raw(reader, labels.data(), labels.size());
        read_raw(reader, observations.data(), observations.size());
        read_raw(reader, relations.data(), relations.size());
        column.load_rows(live, indices.data(), weights.data(), labels.data(),
                         observations.data(), relations.data());
        column.set_births(births);
    }

    template <typename T>
    static void write_raw(std::vector<uint8_t>& out, const T* data, size_t count) {
        if (!count) return;
        const size_t at = out.size();
        out.resize(at + count * sizeof(T));
        std::memcpy(out.data() + at, data, count * sizeof(T));
    }
    template <typename T>
    static void read_raw(Reader& reader, T* data, size_t count) {
        if (!count) return;
        need(reader, count * sizeof(T));
        std::memcpy(data, reader.data + reader.offset, count * sizeof(T));
        reader.offset += count * sizeof(T);
    }
};

}  // namespace alm
