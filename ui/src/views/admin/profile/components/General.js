import {Button, Flex, Icon, Progress, SimpleGrid, Text, useColorModeValue} from "@chakra-ui/react";
import Card from "components/card/Card.js";
import React from "react";
import Information from "views/admin/profile/components/Information";
import {
  MdAdd,
  MdAutorenew,
  MdCheckCircle,
  MdEvent, MdThumbDown, MdThumbUp, MdVerifiedUser,
} from "react-icons/md";
import axios from "redaxios";
// Assets
export default function GeneralInformation(props) {
  const { selectedProject, ...rest } = props;
  // Chakra Color Mode
  const textColorPrimary = useColorModeValue("secondaryGray.900", "white");
  const textColorSecondary = "gray.400";
  const cardShadow = useColorModeValue(
    "0px 18px 40px rgba(112, 144, 176, 0.12)",
    "unset"
  );
  const downloadFile = () => {
    axios.get(`http://localhost:5000/api/get_pdf/project.pdf`, {responseType: 'blob'}).then((response) => {
      window.open(URL.createObjectURL(response.data));
    })
  }

  const getStatusColor = (projectStatus) => {
    if (projectStatus == "Planificat") {
      return "orange.500";
    }

    if (projectStatus == "Aprobat") {
      return "yellow.500";
    }

    if (projectStatus == "În desfășurare") {
      return "red.500";
    }

    return "green.500";
  }

  const getStatusIcon = (projectStatus) => {
    if (projectStatus == "Planificat") {
      return MdEvent;
    }

    if (projectStatus == "Aprobat") {
      return MdVerifiedUser;
    }

    if (projectStatus == "În desfășurare") {
      return MdAutorenew;
    }

    return MdCheckCircle;
  }

  const formatAsMoney = (number) => {
    const formattedNumber = Number(number).toFixed(2); // Rounds to 2 decimal places
    return formattedNumber.replace(/\B(?=(\d{3})+(?!\d))/g, ',');
  };

  return (
    <Card mb={{ base: "0px", "2xl": "20px" }} {...rest}>
        {
            selectedProject && (<>
                <Text
                    color={textColorPrimary}
                    fontWeight='bold'
                    fontSize='2xl'
                    mt='10px'
                    mb='4px'>
                    {selectedProject.project_name}
                </Text>
                  <Text color={textColorSecondary} fontSize='md' me='26px' mb="20px">
                    <Flex align='center'>
                      <Icon
                          w='20px'
                          h='20px'
                          me='3px'
                          color={getStatusColor(selectedProject.project_status)}
                          as={getStatusIcon(selectedProject.project_status)}
                      />
                      <Text fontSize='sm' fontWeight='700'>
                        {selectedProject.project_status} · Categorie: {selectedProject.category_name}
                      </Text>
                    </Flex>
                  </Text>
                <Text color={textColorSecondary} fontSize='md' me='26px' color="#302f2f">
                    {selectedProject.project_description}
                </Text>
            </>
            )
        }
        {selectedProject && selectedProject.project_status == "Aprobat" && (
            <>
              <Text marginTop="30px" mb={1}>Buget acumulat: {formatAsMoney(selectedProject.allocated_budget)} MDL din {formatAsMoney(selectedProject.necessary_budget)} MDL</Text>
              <Progress
                  align='start'
                  colorScheme='brandScheme'
                  value={selectedProject.allocated_budget != 0 ? selectedProject.allocated_budget / selectedProject.necessary_budget * 100 : 0}
                  w='100%'
                  backgroundColor="#E2E8F0"
                  hasStripe={true}
                  isAnimated={true}
              />
            </>
        )}

        {selectedProject && (selectedProject.project_status == "În desfășurare" || selectedProject.project_status == "Finalizat") && (
          <SimpleGrid columns='2' gap='20px' marginTop="30px">
              {selectedProject && selectedProject.expenses.map(e => (
                  <Information
                      boxShadow={cardShadow}
                      title={e.expense_name}
                      fileName={e.expense_pdf_file_name}
                      value={formatAsMoney(e.expense_amount) + " MDL"}
                  />
              ))}
          </SimpleGrid>
        )}

      <Flex justifyContent={"space-between"} marginTop="50px">
        {selectedProject && (selectedProject.project_status != "Planificat") && (
            // <Icon cursor="pointer" w='28px' h='28px' position="absolute" bottom='10px' left='14px' as={MdOutlinePictureAsPdf} color={"#422AFB"} onClick={() => downloadFile()} />
            <Button
                w='140px'
                minW='140px'
                variant='brand'
                fontWeight='500'
                onClick={() => downloadFile()}>
              Descarcă raport
            </Button>
        )}
        {selectedProject && (selectedProject.project_status == "Aprobat") && (
            // <Icon cursor="pointer" w='28px' h='28px' position="absolute" bottom='10px' left='14px' as={MdOutlinePictureAsPdf} color={"#422AFB"} onClick={() => downloadFile()} />
            <Button
                backgroundColor={"green.500"}
                w='140px'
                minW='140px'
                variant='brand'
                fontWeight='500'
                _hover={{ bg: "green.400" }}

                onClick={() => downloadFile()}>
              Donează <Icon
                transition='0.2s linear'
                w='20px'
                h='20px'
                as={MdAdd}
            />
            </Button>
        )}
        {selectedProject && (selectedProject.project_status == "Planificat" || selectedProject.project_status == "Finalizat") && (
              <div>
                <Button
                    backgroundColor="#F4F7FE"
                    _hover={{ bg: "whiteAlpha.900" }}
                    w='40px'
                    h='40px'
                    borderRadius={50}
                    variant='brand'
                    fontWeight='500'
                    onClick={() => downloadFile()}>
                  <Icon
                      transition='0.2s linear'
                      w='20px'
                      h='20px'
                      as={MdThumbUp}
                      color='green.500'
                  />
                </Button>
                <Button
                    backgroundColor="#F4F7FE"
                    _hover={{ bg: "whiteAlpha.900" }}
                    w='40px'
                    h='40px'
                    borderRadius={50}
                    variant='brand'
                    fontWeight='500'
                    marginLeft="10px"
                    onClick={() => downloadFile()}>
                  <Icon
                      transition='0.2s linear'
                      w='20px'
                      h='20px'
                      as={MdThumbDown}
                      color='red.500'
                  />
                </Button>
              </div>)}
      </Flex>
    </Card>
  );
}
