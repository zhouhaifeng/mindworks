// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_model_repo.hpp"

#include <common_test_utils/file_utils.hpp>

namespace TestDataHelpers {
std::string generate_test_xml_file() {
    // Create the file
    std::string plugin_xml = "plugin_test.xml";
    std::ofstream plugin_xml_file(plugin_xml);

    // Write to the file
    plugin_xml_file << "<!--\n";
    plugin_xml_file << "Copyright (C) 2023 Intel Corporation\n";
    plugin_xml_file << "SPDX-License-Identifier: Apache-2.0\n";
    plugin_xml_file << "-->\n";
    plugin_xml_file << "\n";

    plugin_xml_file << "<ie>\n";
    plugin_xml_file << "    <plugins>\n";
    plugin_xml_file << "        <plugin location=\"";
    plugin_xml_file << CommonTestUtils::getExecutableDirectory();
    plugin_xml_file << ov::util::FileTraits<char>::file_separator;
    plugin_xml_file << ov::util::FileTraits<char>::library_prefix();
    plugin_xml_file << "mock_engine";
    plugin_xml_file << IE_BUILD_POSTFIX;
    plugin_xml_file << ov::util::FileTraits<char>::dot_symbol;
    plugin_xml_file << ov::util::FileTraits<char>::library_ext();
    plugin_xml_file << "\" name=\"CUSTOM\">\n";
    plugin_xml_file << "        </plugin>\n";
    plugin_xml_file << "    </plugins>\n";
    plugin_xml_file << "</ie>\n";

    // Close the file
    plugin_xml_file.close();
    return plugin_xml;
}
}  // namespace TestDataHelpers
