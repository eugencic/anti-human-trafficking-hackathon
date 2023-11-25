// Chakra imports
import {Box, Flex, Icon, Text, useColorModeValue} from "@chakra-ui/react";
// Custom components
import Card from "components/card/Card.js";
import React from "react";
import {MdFileCopy, MdOutlinePictureAsPdf, MdPictureAsPdf} from "react-icons/md";
import axios from "redaxios";

export default function Information(props) {
  const { title, value, fileName, ...rest } = props;
  // Chakra Color Mode
  const textColorPrimary = useColorModeValue("secondaryGray.900", "white");
  const textColorSecondary = "gray.400";
  const bg = useColorModeValue("white", "navy.700");

  const downloadFile = () => {
    axios.get(`http://localhost:5000/api/get_pdf/project.pdf`, {responseType: 'blob'}).then((response) => {
      window.open(URL.createObjectURL(response.data));
    })
  }

  return (
      <Card bg={bg} {...rest}>
        <Box>
          <Flex justifyContent="space-between" _hover={{cursor: "pointer"}}>
            <Text fontWeight='500' color={textColorSecondary} fontSize='sm'>
              {title}
            </Text>
            <Icon w='28px' h='28px'  as={MdOutlinePictureAsPdf} color={"#422AFB"} onClick={() => downloadFile()} />
          </Flex>
          <Text color={textColorPrimary} fontWeight='500' fontSize='md'>
            {value}
          </Text>
        </Box>
      </Card>
  );
}
